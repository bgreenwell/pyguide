use pyo3::prelude::*;
use numpy::{PyArrayMethods, PyArray1, PyArray2};
use ndarray::{Array1, Array2};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::BTreeSet;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_chi2, m)?)?;
    m.add_function(wrap_pyfunction!(compute_contingency_table, m)?)?;
    m.add_function(wrap_pyfunction!(bin_continuous, m)?)?;
    m.add_function(wrap_pyfunction!(find_best_threshold_numerical, m)?)?;
    Ok(())
}

#[pyfunction]
fn calculate_chi2(contingency: &Bound<'_, PyArray2<f64>>) -> PyResult<(f64, f64)> {
    let array = unsafe { contingency.as_array() };
    
    let row_sums: Vec<f64> = array.sum_axis(ndarray::Axis(1)).to_vec();
    let col_sums: Vec<f64> = array.sum_axis(ndarray::Axis(0)).to_vec();
    let total: f64 = row_sums.iter().sum();

    if total == 0.0 {
        return Ok((0.0, 0.0));
    }

    let mut chi2_stat = 0.0;
    for (i, &r_sum) in row_sums.iter().enumerate() {
        for (j, &c_sum) in col_sums.iter().enumerate() {
            let expected = (r_sum * c_sum) / total;
            if expected > 0.0 {
                let observed = array[[i, j]];
                chi2_stat += (observed - expected).powi(2) / expected;
            }
        }
    }

    let dof = (row_sums.len() - 1) * (col_sums.len() - 1);
    
    Ok((chi2_stat, dof as f64))
}

#[pyfunction]
fn compute_contingency_table(
    py: Python<'_>,
    x: &Bound<'_, PyArray1<i64>>,
    z: &Bound<'_, PyArray1<i64>>,
    num_x: usize,
    num_z: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let x_view = unsafe { x.as_array() };
    let z_view = unsafe { z.as_array() };
    
    let mut table = Array2::<f64>::zeros((num_x, num_z));
    
    for (&xi, &zi) in x_view.iter().zip(z_view.iter()) {
        if xi >= 0 && (xi as usize) < num_x && zi >= 0 && (zi as usize) < num_z {
            table[[xi as usize, zi as usize]] += 1.0;
        }
    }
    
    Ok(PyArray2::from_array(py, &table).unbind())
}

fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    if p <= 0.0 {
        return sorted_data[0];
    }
    if p >= 100.0 {
        return sorted_data[sorted_data.len() - 1];
    }

    let idx = (p / 100.0) * (sorted_data.len() - 1) as f64;
    let i = idx.floor() as usize;
    let fraction = idx - i as f64;

    if i + 1 < sorted_data.len() {
        sorted_data[i] + (sorted_data[i + 1] - sorted_data[i]) * fraction
    } else {
        sorted_data[i]
    }
}

#[pyfunction]
fn bin_continuous(
    py: Python<'_>,
    x: &Bound<'_, PyArray1<f64>>,
    n_bins: usize,
) -> PyResult<Py<PyArray1<i64>>> {
    let x_view = unsafe { x.as_array() };
    let n = x_view.len();
    if n == 0 {
        return Ok(PyArray1::from_array(py, &Array1::<i64>::zeros(0)).unbind());
    }

    let mut unique_set = BTreeSet::new();
    let mut non_nan_values = Vec::with_capacity(n);
    for &val in x_view.iter() {
        if !val.is_nan() {
            unique_set.insert(ordered_float::OrderedFloat(val));
            non_nan_values.push(val);
        }
    }
    let unique_values: Vec<f64> = unique_set.into_iter().map(|v| v.0).collect();

    if unique_values.is_empty() {
        return Ok(PyArray1::from_array(py, &Array1::<i64>::zeros(n)).unbind());
    }

    if unique_values.len() <= n_bins {
        let mut binned = Array1::<i64>::zeros(n);
        for (i, &val) in x_view.iter().enumerate() {
            if val.is_nan() {
                binned[i] = -1;
            } else if let Ok(idx) = unique_values.binary_search_by(|v| v.partial_cmp(&val).unwrap()) {
                binned[i] = idx as i64;
            }
        }
        return Ok(PyArray1::from_array(py, &binned).unbind());
    }

    non_nan_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut thresholds = Vec::with_capacity(n_bins - 1);
    for i in 1..n_bins {
        let p = (i as f64 * 100.0) / n_bins as f64;
        thresholds.push(percentile(&non_nan_values, p));
    }
    thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let mut binned = Array1::<i64>::zeros(n);
    for (i, &val) in x_view.iter().enumerate() {
        if val.is_nan() {
            binned[i] = -1;
        } else {
            let mut bin = 0;
            for &t in &thresholds {
                if val >= t {
                    bin += 1;
                } else {
                    break;
                }
            }
            binned[i] = bin;
        }
    }

    Ok(PyArray1::from_array(py, &binned).unbind())
}

#[pyfunction]
fn find_best_threshold_numerical(
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    criterion: String,
) -> PyResult<(Option<f64>, bool, f64)> {
    let x_view = unsafe { x.as_array() };
    let y_view = unsafe { y.as_array() };
    let n_total = x_view.len();

    let mut non_nan_data: Vec<(f64, f64)> = Vec::with_capacity(n_total);
    let mut nan_y: Vec<f64> = Vec::new();

    for (&xi, &yi) in x_view.iter().zip(y_view.iter()) {
        if xi.is_nan() {
            nan_y.push(yi);
        } else {
            non_nan_data.push((xi, yi));
        }
    }

    if non_nan_data.is_empty() {
        return Ok((None, true, 0.0));
    }

    // Sort by x
    non_nan_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut best_threshold = None;
    let mut best_gain = -1.0;
    let mut best_missing_go_left = true;

    if criterion == "gini" {
        // Gini logic
        let mut y_int: Vec<usize> = y_view.iter().map(|&yi| yi as usize).collect();
        let n_classes = y_int.iter().max().map_or(0, |&m| m + 1);
        
        let mut total_counts = vec![0.0; n_classes];
        for &yi in &y_int {
            total_counts[yi] += 1.0;
        }
        
        let mut nan_counts = vec![0.0; n_classes];
        for &yi in &nan_y {
            nan_counts[yi as usize] += 1.0;
        }
        
        let current_impurity = 1.0 - total_counts.iter().map(|&c| (c / n_total as f64).powi(2)).sum::<f64>();
        
        let mut left_counts = vec![0.0; n_classes];
        let n_nan = nan_y.len() as f64;
        let n_total_f = n_total as f64;

        for i in 0..non_nan_data.len() - 1 {
            let (_, yi) = non_nan_data[i];
            left_counts[yi as usize] += 1.0;
            
            // Potential split point if next x is different
            if non_nan_data[i].0 < non_nan_data[i+1].0 {
                let n_l_nn = (i + 1) as f64;
                let n_r_nn = (non_nan_data.len() - (i + 1)) as f64;
                
                // Option 1: Missing go left
                let n_l = n_l_nn + n_nan;
                let n_r = n_r_nn;
                if n_l > 0.0 && n_r > 0.0 {
                    let imp_l = 1.0 - left_counts.iter().zip(&nan_counts)
                        .map(|(&lc, &nc)| ((lc + nc) / n_l).powi(2)).sum::<f64>();
                    let imp_r = 1.0 - total_counts.iter().zip(&left_counts).zip(&nan_counts)
                        .map(|((&tc, &lc), &nc)| ((tc - lc - nc) / n_r).powi(2)).sum::<f64>();
                    let gain = current_impurity - (n_l/n_total_f * imp_l + n_r/n_total_f * imp_r);
                    if gain > best_gain {
                        best_gain = gain;
                        best_missing_go_left = true;
                        best_threshold = Some((non_nan_data[i].0 + non_nan_data[i+1].0) / 2.0);
                    }
                }

                // Option 2: Missing go right
                let n_l = n_l_nn;
                let n_r = n_r_nn + n_nan;
                if n_l > 0.0 && n_r > 0.0 {
                    let imp_l = 1.0 - left_counts.iter().map(|&lc| (lc / n_l).powi(2)).sum::<f64>();
                    let imp_r = 1.0 - total_counts.iter().zip(&left_counts)
                        .map(|(&tc, &lc)| ((tc - lc) / n_r).powi(2)).sum::<f64>();
                    let gain = current_impurity - (n_l/n_total_f * imp_l + n_r/n_total_f * imp_r);
                    if gain > best_gain {
                        best_gain = gain;
                        best_missing_go_left = false;
                        best_threshold = Some((non_nan_data[i].0 + non_nan_data[i+1].0) / 2.0);
                    }
                }
            }
        }
    } else {
        // SSE logic
        let sum_y_total: f64 = y_view.iter().sum();
        let sum_y2_total: f64 = y_view.iter().map(|&yi| yi.powi(2)).sum();
        let sum_y_nan: f64 = nan_y.iter().sum();
        let sum_y2_nan: f64 = nan_y.iter().map(|&yi| yi.powi(2)).sum();
        
        let current_impurity = sum_y2_total - (sum_y_total.powi(2) / n_total as f64);
        
        let mut sum_y_l_nn = 0.0;
        let mut sum_y2_l_nn = 0.0;
        let n_nan = nan_y.len() as f64;
        let n_total_f = n_total as f64;

        for i in 0..non_nan_data.len() - 1 {
            let (_, yi) = non_nan_data[i];
            sum_y_l_nn += yi;
            sum_y2_l_nn += yi.powi(2);
            
            if non_nan_data[i].0 < non_nan_data[i+1].0 {
                let n_l_nn = (i + 1) as f64;
                let n_r_nn = (non_nan_data.len() - (i + 1)) as f64;
                
                let sum_y_r_nn = sum_y_total - sum_y_nan - sum_y_l_nn;
                let sum_y2_r_nn = sum_y2_total - sum_y2_nan - sum_y2_l_nn;

                // Option 1: Missing go left
                let n_l = n_l_nn + n_nan;
                let n_r = n_r_nn;
                if n_l > 0.0 && n_r > 0.0 {
                    let imp_l = (sum_y2_l_nn + sum_y2_nan) - (sum_y_l_nn + sum_y_nan).powi(2) / n_l;
                    let imp_r = sum_y2_r_nn - sum_y_r_nn.powi(2) / n_r;
                    let gain = current_impurity - (imp_l + imp_r);
                    if gain > best_gain {
                        best_gain = gain;
                        best_missing_go_left = true;
                        best_threshold = Some((non_nan_data[i].0 + non_nan_data[i+1].0) / 2.0);
                    }
                }

                // Option 2: Missing go right
                let n_l = n_l_nn;
                let n_r = n_r_nn + n_nan;
                if n_l > 0.0 && n_r > 0.0 {
                    let imp_l = sum_y2_l_nn - sum_y_l_nn.powi(2) / n_l;
                    let imp_r = (sum_y2_r_nn + sum_y2_nan) - (sum_y_r_nn + sum_y_nan).powi(2) / n_r;
                    let gain = current_impurity - (imp_l + imp_r);
                    if gain > best_gain {
                        best_gain = gain;
                        best_missing_go_left = false;
                        best_threshold = Some((non_nan_data[i].0 + non_nan_data[i+1].0) / 2.0);
                    }
                }
            }
        }
    }

    Ok((best_threshold, best_missing_go_left, best_gain))
}