#include "io/h5ad_reader.hpp"

#include <stdexcept>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <type_traits>

extern "C" {
#include <hdf5.h>
}

namespace rarecell {

    // ============================ HDF5 helpers ============================

    static inline void h5_check(herr_t status, const char* msg) {
        if (status < 0) throw std::runtime_error(std::string("HDF5 error: ") + msg);
    }

    static bool object_exists(hid_t loc, const char* path) {
        htri_t exists = H5Lexists(loc, path, H5P_DEFAULT);
        return exists > 0;
    }

    static inline size_t safe_strnlen(const char* s, size_t maxlen) {
        size_t n = 0;
        while (n < maxlen && s[n] != '\0') ++n;
        return n;
    }

    // Open file with MPI-IO if available; otherwise serial open.
    static hid_t open_file_with_mpi(const std::string& path, MPI_Comm comm) {
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
#ifdef H5_HAVE_PARALLEL
        MPI_Info info = MPI_INFO_NULL;
        h5_check(H5Pset_fapl_mpio(fapl, comm, info), "H5Pset_fapl_mpio");
#endif
        hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        if (file < 0) throw std::runtime_error("Failed to open H5AD file: " + path);
        return file;
    }

    // Return length of a 1D dataset (no read).
    static int64_t dataset_len_1d(hid_t file, const char* path) {
        hid_t dset = H5Dopen(file, path, H5P_DEFAULT);
        if (dset < 0) throw std::runtime_error(std::string("Dataset not found: ") + path);
        hid_t sp = H5Dget_space(dset);
        int nd = H5Sget_simple_extent_ndims(sp);
        if (nd != 1) {
            H5Sclose(sp); H5Dclose(dset);
            throw std::runtime_error(std::string("Expected 1D dataset: ") + path);
        }
        hsize_t dim;
        H5Sget_simple_extent_dims(sp, &dim, nullptr);
        H5Sclose(sp);
        H5Dclose(dset);
        return static_cast<int64_t>(dim);
    }

    // Do NOT rely on /X attributes; verify CSR by structure.
    static void ensure_csr_matrix(hid_t file) {
        if (!object_exists(file, "/X"))
            throw std::runtime_error("Missing /X in H5AD");

        // /X must be a group containing CSR components
        hid_t obj = H5Oopen(file, "/X", H5P_DEFAULT);
        if (obj < 0) throw std::runtime_error("Failed to open /X");
        H5I_type_t itype = H5Iget_type(obj);
        H5Oclose(obj);

        if (itype != H5I_GROUP) {
            throw std::runtime_error("/X is not a group (dense dataset found). Please write X as CSR.");
        }

        // Require core CSR datasets (shape is optional)
        const char* req[] = { "/X/indptr", "/X/indices", "/X/data" };
        for (auto p : req) {
            if (!object_exists(file, p)) {
                throw std::runtime_error(std::string("Missing required dataset: ") + p);
            }
        }

        // Sanity check that indptr is 1D and has at least 1 element.
        hid_t dset_indptr = H5Dopen(file, "/X/indptr", H5P_DEFAULT);
        hid_t sp = H5Dget_space(dset_indptr);
        int ndims = H5Sget_simple_extent_ndims(sp);
        hsize_t dims[1] = { 0 };
        H5Sget_simple_extent_dims(sp, dims, nullptr);
        H5Sclose(sp);
        H5Dclose(dset_indptr);
        if (ndims != 1 || dims[0] < 1) {
            throw std::runtime_error("/X/indptr must be a 1D array with length >= 1");
        }
    }

    // Try to read shape from /X/shape dataset.
    static bool try_read_shape_dataset(hid_t file, int64_t& n_rows, int64_t& n_cols) {
        if (!object_exists(file, "/X/shape")) return false;
        hid_t dset = H5Dopen(file, "/X/shape", H5P_DEFAULT);
        if (dset < 0) return false;
        int64_t shape[2] = { 0,0 };
        hid_t dtype = H5Dget_type(dset);
        H5T_class_t cls = H5Tget_class(dtype);
        hid_t mtype = (cls == H5T_INTEGER && H5Tget_size(dtype) == 8) ? H5T_NATIVE_LLONG : H5T_NATIVE_INT;
        herr_t st = H5Dread(dset, mtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, shape);
        H5Tclose(dtype);
        H5Dclose(dset);
        if (st < 0) return false;
        n_rows = shape[0]; n_cols = shape[1];
        return true;
    }

    // Try to read shape from /X group's numeric attribute "shape" (2 integers).
    static bool try_read_shape_attribute(hid_t file, int64_t& n_rows, int64_t& n_cols) {
        hid_t obj = H5Oopen(file, "/X", H5P_DEFAULT);
        if (obj < 0) return false;
        if (H5Aexists(obj, "shape") <= 0) { H5Oclose(obj); return false; }
        hid_t attr = H5Aopen(obj, "shape", H5P_DEFAULT);
        if (attr < 0) { H5Oclose(obj); return false; }

        hid_t atype = H5Aget_type(attr);
        if (H5Tget_class(atype) != H5T_INTEGER) {
            H5Tclose(atype); H5Aclose(attr); H5Oclose(obj);
            return false;
        }

        // Expect 1D, length >= 2
        hid_t aspace = H5Aget_space(attr);
        int nd = H5Sget_simple_extent_ndims(aspace);
        hsize_t dims[1] = { 0 };
        H5Sget_simple_extent_dims(aspace, dims, nullptr);

        int ok = (nd == 1 && dims[0] >= 2);
        int64_t tmp[2] = { 0,0 };
        if (ok) {
            // Use native 64-bit; HDF5 will convert
            herr_t st = H5Aread(attr, H5T_NATIVE_LLONG, tmp);
            if (st < 0) ok = 0;
        }

        H5Sclose(aspace);
        H5Tclose(atype);
        H5Aclose(attr);
        H5Oclose(obj);

        if (!ok) return false;
        n_rows = tmp[0]; n_cols = tmp[1];
        return true;
    }

    // Deduce n_cols from /var: prefer /var/_index length; else /var/Gene (dataset or codes).
    static int64_t deduce_ncols(hid_t file) {
        if (object_exists(file, "/var/_index")) return dataset_len_1d(file, "/var/_index");
        if (object_exists(file, "/var/Gene/codes")) return dataset_len_1d(file, "/var/Gene/codes");
        if (object_exists(file, "/var/Gene")) return dataset_len_1d(file, "/var/Gene");
        throw std::runtime_error("Cannot determine number of genes: missing /var/_index and /var/Gene.");
    }

    // Read overall matrix shape with multiple fallbacks.
    static std::pair<int64_t, int64_t> read_shape(hid_t file) {
        int64_t n_rows = -1, n_cols = -1;

        // 1) /X/shape dataset
        if (try_read_shape_dataset(file, n_rows, n_cols)) return { n_rows, n_cols };
        // 2) /X attribute "shape"
        if (try_read_shape_attribute(file, n_rows, n_cols)) return { n_rows, n_cols };
        // 3) Deduce: rows from len(indptr)-1; cols from /var
        int64_t indptr_len = dataset_len_1d(file, "/X/indptr");
        if (indptr_len < 1) throw std::runtime_error("/X/indptr has invalid length");
        n_rows = indptr_len - 1;
        n_cols = deduce_ncols(file);
        return { n_rows, n_cols };
    }

    template <typename T>
    static void read_1d_slice_numeric(hid_t dset, int64_t offset, int64_t count, std::vector<T>& out) {
        out.resize(static_cast<size_t>(count));
        hid_t fspace = H5Dget_space(dset);
        hsize_t start[1] = { static_cast<hsize_t>(offset) };
        hsize_t stride[1] = { 1 };
        hsize_t count_h[1] = { static_cast<hsize_t>(count) };
        hsize_t block[1] = { 1 };
        h5_check(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, stride, count_h, block), "select hyperslab");

        hid_t mspace = H5Screate_simple(1, count_h, nullptr);

        hid_t mtype = -1;
        if (std::is_integral<T>::value) {
            if (sizeof(T) == 8) mtype = H5T_NATIVE_LLONG;
            else if (sizeof(T) == 4) mtype = H5T_NATIVE_INT;
            else if (sizeof(T) == 2) mtype = H5T_NATIVE_SHORT;
            else if (sizeof(T) == 1) mtype = H5T_NATIVE_CHAR;
        }
        else if (std::is_floating_point<T>::value) {
            if (sizeof(T) == 8) mtype = H5T_NATIVE_DOUBLE;
            else if (sizeof(T) == 4) mtype = H5T_NATIVE_FLOAT;
        }
        if (mtype < 0) { H5Sclose(mspace); H5Sclose(fspace); throw std::runtime_error("Unsupported numeric dtype for read_1d_slice_numeric"); }

        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
#ifdef H5_HAVE_PARALLEL
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);
#endif
        h5_check(H5Dread(dset, mtype, mspace, fspace, dxpl, out.data()), "H5Dread numeric slice");
        H5Pclose(dxpl);

        H5Sclose(mspace);
        H5Sclose(fspace);
    }

    static std::vector<int64_t> read_indptr_slice(hid_t file, int64_t row0, int64_t row1) {
        // Need [row0 : row1] plus terminal at row1
        int64_t count = (row1 - row0) + 1;
        hid_t dset = H5Dopen(file, "/X/indptr", H5P_DEFAULT);
        if (dset < 0) throw std::runtime_error("Missing /X/indptr");
        std::vector<int64_t> slice;
        read_1d_slice_numeric<int64_t>(dset, row0, count, slice);
        H5Dclose(dset);
        return slice;
    }

    static void read_indices_data_slices(hid_t file, int64_t nnz0, int64_t nnz,
        std::vector<int32_t>& idx, std::vector<float>& val) {
        // indices
            {
                hid_t dset = H5Dopen(file, "/X/indices", H5P_DEFAULT);
                if (dset < 0) throw std::runtime_error("Missing /X/indices");
                read_1d_slice_numeric<int32_t>(dset, nnz0, nnz, idx);
                H5Dclose(dset);
            }
            // data
            {
                hid_t dset = H5Dopen(file, "/X/data", H5P_DEFAULT);
                if (dset < 0) throw std::runtime_error("Missing /X/data");
                read_1d_slice_numeric<float>(dset, nnz0, nnz, val);
                H5Dclose(dset);
            }
    }

    // Read a 1D string dataset slice (supports variable-length and fixed-length)
    static std::vector<std::string>
        read_strings_range(hid_t file, const std::string& dset_path,
            int64_t offset, int64_t count) {
        hid_t dset = H5Dopen(file, dset_path.c_str(), H5P_DEFAULT);
        if (dset < 0) throw std::runtime_error("Dataset not found: " + dset_path);

        hid_t fspace = H5Dget_space(dset);
        hsize_t start[1] = { static_cast<hsize_t>(offset) };
        hsize_t cnt[1] = { static_cast<hsize_t>(count) };
        hsize_t stride[1] = { 1 }, block[1] = { 1 };
        h5_check(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, stride, cnt, block),
            "select string hyperslab");

        hid_t mspace = H5Screate_simple(1, cnt, nullptr);
        hid_t ftype = H5Dget_type(dset);
        if (H5Tget_class(ftype) != H5T_STRING) {
            H5Tclose(ftype); H5Sclose(mspace); H5Sclose(fspace); H5Dclose(dset);
            throw std::runtime_error("Dataset is not a string array: " + dset_path);
        }

        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
#ifdef H5_HAVE_PARALLEL
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);
#endif

        std::vector<std::string> out;
        out.reserve(static_cast<size_t>(count));

        if (H5Tis_variable_str(ftype) > 0) {
            std::vector<char*> tmp(static_cast<size_t>(count), nullptr);
            hid_t mtype = H5Tcopy(H5T_C_S1);
            H5Tset_size(mtype, H5T_VARIABLE);
            H5Tset_cset(mtype, H5Tget_cset(ftype));
            H5Tset_strpad(mtype, H5Tget_strpad(ftype));
            h5_check(H5Dread(dset, mtype, mspace, fspace, dxpl, tmp.data()), "H5Dread vlen strings");
            for (int64_t i = 0; i < count; ++i) out.emplace_back(tmp[size_t(i)] ? tmp[size_t(i)] : "");
            h5_check(H5Dvlen_reclaim(mtype, mspace, H5P_DEFAULT, tmp.data()), "H5Dvlen_reclaim");
            H5Tclose(mtype);
        }
        else {
            const size_t sz = H5Tget_size(ftype);
            std::vector<char> buf(static_cast<size_t>(count) * sz, 0);
            hid_t mtype = H5Tcopy(H5T_C_S1);
            H5Tset_size(mtype, sz);
            H5Tset_cset(mtype, H5Tget_cset(ftype));
            H5Tset_strpad(mtype, H5Tget_strpad(ftype));
            h5_check(H5Dread(dset, mtype, mspace, fspace, dxpl, buf.data()), "H5Dread fixed-len strings");
            for (int64_t i = 0; i < count; ++i) {
                const char* s = buf.data() + size_t(i) * sz;
                std::string v(s, safe_strnlen(s, sz));
                while (!v.empty() && (v.back() == ' ')) v.pop_back(); // trim SPACEPAD
                out.emplace_back(std::move(v));
            }
            H5Tclose(mtype);
        }

        H5Pclose(dxpl);
        H5Tclose(ftype);
        H5Sclose(mspace);
        H5Sclose(fspace);
        H5Dclose(dset);
        return out;
    }

    static std::vector<std::string>
        read_strings_full(hid_t file, const std::string& dset_path) {
        hid_t dset = H5Dopen(file, dset_path.c_str(), H5P_DEFAULT);
        if (dset < 0) throw std::runtime_error("Dataset not found: " + dset_path);
        hid_t sp = H5Dget_space(dset);
        if (H5Sget_simple_extent_ndims(sp) != 1) {
            H5Sclose(sp); H5Dclose(dset);
            throw std::runtime_error("Expected 1D dataset: " + dset_path);
        }
        hsize_t dim; H5Sget_simple_extent_dims(sp, &dim, nullptr);
        H5Sclose(sp); H5Dclose(dset);
        return read_strings_range(file, dset_path, 0, static_cast<int64_t>(dim));
    }

    // Robust attribute-string reader (handles fixed- and variable-length; preserves cset/padding)
    static std::string read_attr_string_safe(hid_t obj, const char* attr_name) {
        if (H5Aexists(obj, attr_name) <= 0) return {};
        hid_t attr = H5Aopen(obj, attr_name, H5P_DEFAULT);
        if (attr < 0) return {};
        hid_t ftype = H5Aget_type(attr);
        if (H5Tget_class(ftype) != H5T_STRING) {
            H5Tclose(ftype); H5Aclose(attr);
            return {};
        }

        std::string out;
        if (H5Tis_variable_str(ftype) > 0) {
            char* s = nullptr;
            hid_t mtype = H5Tcopy(H5T_C_S1);
            H5Tset_size(mtype, H5T_VARIABLE);
            H5Tset_cset(mtype, H5Tget_cset(ftype));
            H5Tset_strpad(mtype, H5Tget_strpad(ftype));
            if (H5Aread(attr, mtype, &s) >= 0) out = s ? s : "";
            if (s) H5free_memory(s);
            H5Tclose(mtype);
        }
        else {
            const size_t sz = H5Tget_size(ftype);
            std::vector<char> buf(sz, 0);
            hid_t mtype = H5Tcopy(H5T_C_S1);
            H5Tset_size(mtype, sz);
            H5Tset_cset(mtype, H5Tget_cset(ftype));
            H5Tset_strpad(mtype, H5Tget_strpad(ftype));
            if (H5Aread(attr, mtype, buf.data()) >= 0) {
                out.assign(buf.data(), safe_strnlen(buf.data(), sz));
                while (!out.empty() && out.back() == ' ') out.pop_back();
            }
            H5Tclose(mtype);
        }

        H5Tclose(ftype);
        H5Aclose(attr);
        return out;
    }

    // Read /obs/<col> as strings. Handles: string dataset; integer codes + categories;
    // categorical group with {codes,categories}; and global _categories layout.
    static std::vector<std::string>
        read_obs_column_as_strings(hid_t file, const std::string& col,
            int64_t offset, int64_t count) {
        const std::string base = "/obs/" + col;

        if (object_exists(file, base.c_str())) {
            hid_t obj = H5Oopen(file, base.c_str(), H5P_DEFAULT);
            H5I_type_t t = H5Iget_type(obj);
            H5Oclose(obj);

            if (t == H5I_DATASET) {
                // Try strings directly
                try { return read_strings_range(file, base, offset, count); }
                catch (...) {
                    // Not a string dataset -> may be integer codes with side categories
                    if (object_exists(file, (base + "/categories").c_str())) {
                        std::vector<int32_t> codes;
                        hid_t dset_codes = H5Dopen(file, base.c_str(), H5P_DEFAULT);
                        read_1d_slice_numeric<int32_t>(dset_codes, offset, count, codes);
                        H5Dclose(dset_codes);
                        auto cats = read_strings_full(file, base + "/categories");
                        std::vector<std::string> out; out.reserve(codes.size());
                        for (auto c : codes) out.emplace_back((c >= 0 && c < (int)cats.size()) ? cats[c] : "");
                        return out;
                    }
                    // Global categories
                    const std::string catgrp = "/obs/_categories/" + col + "/categories";
                    if (object_exists(file, catgrp.c_str())) {
                        std::vector<int32_t> codes;
                        hid_t dset_codes = H5Dopen(file, base.c_str(), H5P_DEFAULT);
                        read_1d_slice_numeric<int32_t>(dset_codes, offset, count, codes);
                        H5Dclose(dset_codes);
                        auto cats = read_strings_full(file, catgrp);
                        std::vector<std::string> out; out.reserve(codes.size());
                        for (auto c : codes) out.emplace_back((c >= 0 && c < (int)cats.size()) ? cats[c] : "");
                        return out;
                    }
                    // Fallback: stringify codes
                    std::vector<int32_t> codes;
                    hid_t dset_codes = H5Dopen(file, base.c_str(), H5P_DEFAULT);
                    read_1d_slice_numeric<int32_t>(dset_codes, offset, count, codes);
                    H5Dclose(dset_codes);
                    std::vector<std::string> out; out.reserve(codes.size());
                    for (auto c : codes) out.emplace_back(std::to_string(c));
                    return out;
                }
            }
            else if (t == H5I_GROUP) {
                // Expect /obs/<col>/{codes,categories}
                const std::string codesp = base + "/codes";
                const std::string catsp = base + "/categories";
                if (object_exists(file, codesp.c_str()) && object_exists(file, catsp.c_str())) {
                    std::vector<int32_t> codes;
                    hid_t dset_codes = H5Dopen(file, codesp.c_str(), H5P_DEFAULT);
                    read_1d_slice_numeric<int32_t>(dset_codes, offset, count, codes);
                    H5Dclose(dset_codes);
                    auto cats = read_strings_full(file, catsp);
                    std::vector<std::string> out; out.reserve(codes.size());
                    for (auto c : codes) out.emplace_back((c >= 0 && c < (int)cats.size()) ? cats[c] : "");
                    return out;
                }
            }
        }

        throw std::runtime_error("Could not read /obs/" + col + " as strings.");
    }

    // Read /var/<col> as strings for all genes (n items). Handles strings, codes+categories,
    // group categorical layout, or falls back to /var/_index.
    static std::vector<std::string>
        read_var_column_as_strings(hid_t file, const std::string& col, int64_t n) {
        const std::string base = "/var/" + col;

        if (object_exists(file, base.c_str())) {
            hid_t obj = H5Oopen(file, base.c_str(), H5P_DEFAULT);
            H5I_type_t t = H5Iget_type(obj);
            H5Oclose(obj);

            if (t == H5I_DATASET) {
                try { return read_strings_range(file, base, 0, n); }
                catch (...) {
                    if (object_exists(file, (base + "/categories").c_str())) {
                        std::vector<int32_t> codes;
                        hid_t dset_codes = H5Dopen(file, base.c_str(), H5P_DEFAULT);
                        read_1d_slice_numeric<int32_t>(dset_codes, 0, n, codes);
                        H5Dclose(dset_codes);
                        auto cats = read_strings_full(file, base + "/categories");
                        std::vector<std::string> out; out.reserve(codes.size());
                        for (auto c : codes) out.emplace_back((c >= 0 && c < (int)cats.size()) ? cats[c] : "");
                        return out;
                    }
                    const std::string catgrp = "/var/_categories/" + col + "/categories";
                    if (object_exists(file, catgrp.c_str())) {
                        std::vector<int32_t> codes;
                        hid_t dset_codes = H5Dopen(file, base.c_str(), H5P_DEFAULT);
                        read_1d_slice_numeric<int32_t>(dset_codes, 0, n, codes);
                        H5Dclose(dset_codes);
                        auto cats = read_strings_full(file, catgrp);
                        std::vector<std::string> out; out.reserve(codes.size());
                        for (auto c : codes) out.emplace_back((c >= 0 && c < (int)cats.size()) ? cats[c] : "");
                        return out;
                    }
                    // Fallback: stringify codes
                    std::vector<int32_t> codes;
                    hid_t dset_codes = H5Dopen(file, base.c_str(), H5P_DEFAULT);
                    read_1d_slice_numeric<int32_t>(dset_codes, 0, n, codes);
                    H5Dclose(dset_codes);
                    std::vector<std::string> out; out.reserve(codes.size());
                    for (auto c : codes) out.emplace_back(std::to_string(c));
                    return out;
                }
            }
            else if (t == H5I_GROUP) {
                const std::string codesp = base + "/codes";
                const std::string catsp = base + "/categories";
                if (object_exists(file, codesp.c_str()) && object_exists(file, catsp.c_str())) {
                    std::vector<int32_t> codes;
                    hid_t dset_codes = H5Dopen(file, codesp.c_str(), H5P_DEFAULT);
                    read_1d_slice_numeric<int32_t>(dset_codes, 0, n, codes);
                    H5Dclose(dset_codes);
                    auto cats = read_strings_full(file, catsp);
                    std::vector<std::string> out; out.reserve(codes.size());
                    for (auto c : codes) out.emplace_back((c >= 0 && c < (int)cats.size()) ? cats[c] : "");
                    return out;
                }
            }
        }

        // Fallback to /var/_index if custom column is missing
        if (object_exists(file, "/var/_index")) {
            return read_strings_range(file, "/var/_index", 0, n);
        }

        throw std::runtime_error("Could not read /var/" + col + " (or /var/_index) as strings.");
    }

    // Try to read the obs index (cell IDs) from multiple known locations;
    // if a usable source is found, fill 'out' and return true.
    static bool try_read_obs_index(hid_t file, int64_t offset, int64_t count, std::vector<std::string>& out) {
        const char* candidates[] = {
          "/obs/_index",
          "/obs_names",
          "/obs/obs_names",
          "/obs/index",
          "/obs/__index__",
          "/obs/INDEX"
        };

        // 1) Common dataset locations
        for (const char* p : candidates) {
            if (!object_exists(file, p)) continue;

            // Inspect type class
            hid_t dset = H5Dopen(file, p, H5P_DEFAULT);
            if (dset < 0) continue;
            hid_t dtype = H5Dget_type(dset);
            H5T_class_t cls = H5Tget_class(dtype);
            H5Tclose(dtype);
            H5Dclose(dset);

            try {
                if (cls == H5T_STRING) {
                    out = read_strings_range(file, p, offset, count);
                    return true;
                }
                else if (cls == H5T_INTEGER) {
                    std::vector<long long> vals;
                    dset = H5Dopen(file, p, H5P_DEFAULT);
                    read_1d_slice_numeric<long long>(dset, offset, count, vals);
                    H5Dclose(dset);
                    out.clear(); out.reserve(vals.size());
                    for (auto v : vals) out.emplace_back(std::to_string(v));
                    return true;
                }
                else if (cls == H5T_FLOAT) {
                    std::vector<double> vals;
                    dset = H5Dopen(file, p, H5P_DEFAULT);
                    read_1d_slice_numeric<double>(dset, offset, count, vals);
                    H5Dclose(dset);
                    out.clear(); out.reserve(vals.size());
                    for (auto v : vals) out.emplace_back(std::to_string(v));
                    return true;
                }
            }
            catch (...) {
                // try next candidate
            }
        }

        // 2) Attribute on /obs naming the index column (e.g., "index" or "_index")
        hid_t obs = H5Oopen(file, "/obs", H5P_DEFAULT);
        if (obs >= 0) {
            std::string idxname;
            if (H5Aexists(obs, "index") > 0) idxname = read_attr_string_safe(obs, "index");
            if (idxname.empty() && H5Aexists(obs, "_index") > 0) idxname = read_attr_string_safe(obs, "_index");
            H5Oclose(obs);
            if (!idxname.empty()) {
                try {
                    out = read_obs_column_as_strings(file, idxname, offset, count);
                    return true;
                }
                catch (...) {
                    // ignore; fall through
                }
            }
        }

        return false;
    }

    // ============================ Reader methods ============================

    H5ADReader::H5ADReader(MPI_Comm comm) : comm_(comm) {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &world_);
    }

    void H5ADReader::split_rows(int64_t n_rows, int rank, int world, int64_t& row0, int64_t& row1) {
        int64_t base = n_rows / world;
        int64_t rem = n_rows % world;
        row0 = rank * base + (rank < rem ? rank : rem);
        row1 = row0 + base + (rank < rem ? 1 : 0);
    }

    H5ADReadResult H5ADReader::read(const std::string& path) {
        H5ADReadResult out;

        hid_t file = open_file_with_mpi(path, comm_);
        ensure_csr_matrix(file);

        // Global shape (cells x genes) with fallbacks
        auto [n_rows, n_cols] = read_shape(file);
        out.X_local.n_rows = n_rows;
        out.X_local.n_cols = n_cols;

        // Row partition across ranks
        int64_t row0 = 0, row1 = 0;
        split_rows(n_rows, rank_, world_, row0, row1);
        out.X_local.row0 = row0;
        out.X_local.row1 = row1;

        // Local CSR slice: indptr rebased to 0; contiguous indices/data slice
        std::vector<int64_t> indptr_slice = read_indptr_slice(file, row0, row1);
        const int64_t nnz0 = indptr_slice.front();
        const int64_t nnz1 = indptr_slice.back();
        const int64_t local_nnz = nnz1 - nnz0;

        out.X_local.indptr.resize(static_cast<size_t>(row1 - row0 + 1));
        for (size_t i = 0; i < out.X_local.indptr.size(); ++i) {
            out.X_local.indptr[i] = indptr_slice[i] - nnz0;
        }

        read_indices_data_slices(file, nnz0, local_nnz, out.X_local.indices, out.X_local.data);

        // Gene names: from /var/Gene (like your AnnData), fallback to /var/_index
        const std::string gene_column = "Gene";
        out.gene_names = read_var_column_as_strings(file, gene_column, n_cols);

        // Cell IDs: try multiple encodings; fallback to row numbers if none found
        std::vector<std::string> ids;
        if (try_read_obs_index(file, row0, (row1 - row0), ids)) {
            out.cell_ids_local = std::move(ids);
        }
        else {
            // Fallback: global row numbers as strings
            const int64_t local = (row1 - row0);
            out.cell_ids_local.resize(static_cast<size_t>(local));
            for (int64_t i = 0; i < local; ++i) out.cell_ids_local[size_t(i)] = std::to_string(row0 + i);
            if (rank_ == 0) {
                std::cerr << "[warn] /obs index not found; using global row numbers as cell IDs\n";
            }
        }

        // Cell type labels: /obs/cell_type (string or categorical; local slice)
        const std::string cell_type_col = "cell_type";
        out.cell_type_local = read_obs_column_as_strings(file, cell_type_col, row0, (row1 - row0));

        H5Fclose(file);
        return out;
    }

} // namespace rarecell
