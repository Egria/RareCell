#include "metrics/lowess.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rarecell {

    // ---- Compute global log2(max + 0.1) from RAW counts (kept rows & genes) ----
    static std::vector<double>
        compute_log2max_raw_global(MPI_Comm comm, const H5ADReadResult& R, const FilterOutputs& outs)
    {
        const int64_t G = (int64_t)outs.gene_names_filtered.size();
        std::vector<double> max_local((size_t)G, 0.0);

        const auto& Xin = R.X_local;              // RAW integer counts (float-stored)
        const auto& rows = outs.kept_local_rows;   // local kept rows in RAW matrix space
        const auto& remap = outs.col_remap_final;   // old col -> kept col index

        for (size_t i = 0; i < rows.size(); ++i) {
            int64_t r = rows[i];
            int64_t s = Xin.indptr[(size_t)r];
            int64_t e = Xin.indptr[(size_t)r + 1];
            for (int64_t p = s; p < e; ++p) {
                int32_t oldc = Xin.indices[(size_t)p];
                int32_t newc = remap[(size_t)oldc];
                if (newc < 0) continue;
                double v = (double)Xin.data[(size_t)p];   // RAW counts
                if (v > max_local[(size_t)newc]) max_local[(size_t)newc] = v;
            }
        }

        std::vector<double> max_global((size_t)G, 0.0);
        if (G > 0) {
            MPI_Allreduce(max_local.data(), max_global.data(), (int)G, MPI_DOUBLE, MPI_MAX, comm);
        }
        for (int64_t g = 0; g < G; ++g) max_global[(size_t)g] = std::log2(max_global[(size_t)g] + 0.1);
        return max_global;
    }

    // ---- Numpy-like linear-Interpolated quantile of a vector<double> ----
    static inline double quantile_linear(std::vector<double> a, double q) {
        const size_t n = a.size();
        if (n == 0) return std::numeric_limits<double>::infinity();
        if (q <= 0.0) { auto it = std::min_element(a.begin(), a.end()); return *it; }
        if (q >= 1.0) { auto it = std::max_element(a.begin(), a.end()); return *it; }
        std::sort(a.begin(), a.end());
        double pos = q * (double)(n - 1);
        size_t lo = (size_t)std::floor(pos);
        size_t hi = (size_t)std::ceil(pos);
        double h = pos - (double)lo;
        return (1.0 - h) * a[lo] + h * a[hi];
    }

    // ---- Single-pass LOWESS (local linear, tricube), 1D ----
    static std::vector<double>
        lowess_fit_once(const std::vector<double>& x, const std::vector<double>& y, double frac)
    {
        const size_t n = x.size();
        if (n == 0) return {};
        if (n == 1) return std::vector<double>(1, y[0]);

        // stable sort by x
        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return x[a] < x[b]; });

        std::vector<double> xs(n), ys(n);
        for (size_t i = 0; i < n; ++i) { xs[i] = x[idx[i]]; ys[i] = y[idx[i]]; }

        int k = (int)std::ceil(frac * (double)n);
        if (k < 2) k = 2;
        if (k > (int)n) k = (int)n;

        std::vector<double> yhat_sorted(n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            int i_center = (int)i;
            int s = i_center - k / 2;
            if (s < 0) s = 0;
            if (s > (int)n - k) s = (int)n - k;

            double xi = xs[i];
            double maxd = 0.0;
            for (int j = 0; j < k; ++j) {
                double d = std::fabs(xs[(size_t)(s + j)] - xi);
                if (d > maxd) maxd = d;
            }

            double S0 = 0.0, S1 = 0.0, S2 = 0.0, T0 = 0.0, T1 = 0.0;
            const double eps = 1e-12;

            if (maxd <= 0.0) {
                for (int j = 0; j < k; ++j) {
                    double w = 1.0;
                    double xj = xs[(size_t)(s + j)];
                    double yj = ys[(size_t)(s + j)];
                    S0 += w; S1 += w * xj; S2 += w * xj * xj;
                    T0 += w * yj; T1 += w * xj * yj;
                }
            }
            else {
                for (int j = 0; j < k; ++j) {
                    double xj = xs[(size_t)(s + j)];
                    double yj = ys[(size_t)(s + j)];
                    double u = std::fabs(xj - xi) / maxd;
                    double w = 0.0;
                    if (u < 1.0) { double t = 1.0 - u * u * u; w = t * t * t; } // tricube
                    S0 += w; S1 += w * xj; S2 += w * xj * xj;
                    T0 += w * yj; T1 += w * xj * yj;
                }
            }

            double yhat = 0.0;
            if (S0 <= eps) {
                double ssum = 0.0;
                for (int j = 0; j < k; ++j) ssum += ys[(size_t)(s + j)];
                yhat = ssum / (double)k;
            }
            else {
                double denom = S0 * S2 - S1 * S1;
                if (std::fabs(denom) <= eps) {
                    yhat = T0 / S0;
                }
                else {
                    double beta1 = (S0 * T1 - S1 * T0) / denom;
                    double beta0 = (T0 - beta1 * S1) / S0;
                    yhat = beta0 + beta1 * xi;
                }
            }
            yhat_sorted[i] = yhat;
        }

        // unsort back to original order
        std::vector<double> yhat(n, 0.0);
        for (size_t i = 0; i < n; ++i) yhat[idx[i]] = yhat_sorted[i];
        return yhat;
    }

    // ---- Two-pass LOWESS detrending (generic metric), rank 0 result ----
    std::vector<float> lowess_twopass_detrend_metric_rank0(MPI_Comm comm,
        const H5ADReadResult& R,
        const FilterOutputs& outs,
        const std::vector<float>& metric_rank0,
        double outlier_q,
        double span)
    {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);

        const int64_t G = (int64_t)outs.gene_names_filtered.size();
        if (G == 0) return {};

        // x = log2(max + 0.1) from RAW counts (computed globally)
        std::vector<double> x_log2max = compute_log2max_raw_global(comm, R, outs);

        if (rank != 0) {
            return {}; // only rank 0 computes residuals
        }

        // Validate inputs on rank 0
        if ((int64_t)metric_rank0.size() != G) {
            throw std::runtime_error("lowess_twopass_detrend_metric_rank0: metric size != #kept genes");
        }
        if (!(span > 0.0 && span <= 1.0)) {
            throw std::runtime_error("lowess_twopass_detrend_metric_rank0: span must be in (0,1]");
        }
        if (!(outlier_q > 0.0 && outlier_q < 1.0)) {
            throw std::runtime_error("lowess_twopass_detrend_metric_rank0: outlier quantile must be in (0,1)");
        }

        // Prepare (x, y)
        std::vector<double> x((size_t)G), y((size_t)G);
        for (size_t g = 0; g < (size_t)G; ++g) {
            x[g] = x_log2max[g];
            y[g] = (double)metric_rank0[g];
        }

        // Pass 1
        std::vector<double> f1 = lowess_fit_once(x, y, span);
        std::vector<double> r1((size_t)G);
        for (size_t g = 0; g < (size_t)G; ++g) r1[g] = y[g] - f1[g];

        // Threshold from positive residuals
        std::vector<double> pos;
        pos.reserve((size_t)G);
        for (double v : r1) if (v > 0.0) pos.push_back(v);
        const double thresh = pos.empty() ? std::numeric_limits<double>::infinity()
            : quantile_linear(pos, outlier_q);

        // Inlier mask
        std::vector<uint8_t> inlier((size_t)G, 0);
        size_t nin = 0;
        for (size_t g = 0; g < (size_t)G; ++g) {
            bool keep = (r1[g] < thresh);
            inlier[g] = keep ? 1u : 0u;
            nin += keep ? 1u : 0u;
        }
        if (nin == 0 || nin == (size_t)G) {
            std::vector<float> r2f((size_t)G);
            for (size_t g = 0; g < (size_t)G; ++g) r2f[g] = (float)r1[g];
            return r2f;
        }

        // Build inlier arrays
        std::vector<double> x2; x2.reserve(nin);
        std::vector<double> y2; y2.reserve(nin);
        std::vector<size_t>  map_in; map_in.reserve(nin);
        for (size_t g = 0; g < (size_t)G; ++g) if (inlier[g]) {
            x2.push_back(x[g]); y2.push_back(y[g]); map_in.push_back(g);
        }

        // Pass 2 fit on inliers
        std::vector<double> f2_in = lowess_fit_once(x2, y2, span);

        // Prepare sorted (xs, fs) on inliers
        const size_t n2 = x2.size();
        std::vector<size_t> ord(n2);
        for (size_t i = 0; i < n2; ++i) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return x2[a] < x2[b]; });

        std::vector<double> xs(n2), fs(n2);
        for (size_t i = 0; i < n2; ++i) { xs[i] = x2[ord[i]]; fs[i] = f2_in[ord[i]]; }

        // Interp/extrap helper
        auto predict_on_inlier_curve = [&](double xq)->double {
            if (n2 == 0) return 0.0;
            if (n2 == 1) return fs[0];
            auto it = std::lower_bound(xs.begin(), xs.end(), xq);
            size_t pos = (size_t)std::distance(xs.begin(), it);
            if (pos < n2 && xs[pos] == xq) return fs[pos];
            if (pos == 0) {
                double x0 = xs[0], x1 = xs[1], y0 = fs[0], y1 = fs[1];
                if (x1 == x0) return y0;
                return y1 - (y1 - y0) * (x1 - xq) / (x1 - x0);
            }
            else if (pos >= n2) {
                double x0 = xs[n2 - 2], x1 = xs[n2 - 1], y0 = fs[n2 - 2], y1 = fs[n2 - 1];
                if (x1 == x0) return y1;
                return y0 + (y1 - y0) * (xq - x0) / (x1 - x0);
            }
            else {
                double x0 = xs[pos - 1], x1 = xs[pos], y0 = fs[pos - 1], y1 = fs[pos];
                if (x1 == x0) return 0.5 * (y0 + y1);
                return y0 + (y1 - y0) * (xq - x0) / (x1 - x0);
            }
            };

        // Assemble residuals
        std::vector<double> yhat((size_t)G, std::numeric_limits<double>::quiet_NaN());
        // inliers
        for (size_t k = 0; k < n2; ++k) {
            size_t gi = map_in[k];
            yhat[gi] = f2_in[k];
        }
        // outliers
        for (size_t g = 0; g < (size_t)G; ++g) if (!inlier[g]) {
            yhat[g] = predict_on_inlier_curve(x[g]);
        }

        std::vector<float> r2((size_t)G);
        for (size_t g = 0; g < (size_t)G; ++g) {
            double yh = yhat[g];
            if (!std::isfinite(yh)) yh = f1[g]; // fallback
            r2[g] = (float)(y[g] - yh);
        }
        return r2;
    }

} // namespace rarecell
