#include "config/filter_config.hpp"
#include <fstream>
#include <stdexcept>
#include <algorithm> // std::transform, std::tolower
#include <nlohmann/json.hpp>

namespace rarecell {

    using nlohmann::json;

    // Generic helper: if key exists and is non-null, assign to target (strongly typed).
    template <typename T>
    static void set_if(const json& j, const char* key, T& target) {
        auto it = j.find(key);
        if (it != j.end() && !it->is_null()) {
            it->get_to(target);
        }
    }

    // Overload for float to accept integer/double in JSON without surprises.
    static void set_if_number(const json& j, const char* key, float& target) {
        auto it = j.find(key);
        if (it != j.end() && !it->is_null()) {
            if (it->is_number_float())       target = static_cast<float>(it->get<double>());
            else if (it->is_number_integer()) target = static_cast<float>(it->get<long long>());
            else if (it->is_number_unsigned())target = static_cast<float>(it->get<unsigned long long>());
            else                               it->get_to(target); // let it throw if truly incompatible
        }
    }

    // Overload for int to accept numeric JSON of any kind.
    static void set_if_number(const json& j, const char* key, int& target) {
        auto it = j.find(key);
        if (it != j.end() && !it->is_null()) {
            if (it->is_number_integer())       target = static_cast<int>(it->get<long long>());
            else if (it->is_number_unsigned()) target = static_cast<int>(it->get<unsigned long long>());
            else if (it->is_number_float())    target = static_cast<int>(it->get<double>());
            else                                it->get_to(target);
        }
    }

    FilterConfig load_filter_config(const std::string& json_path) {
        std::ifstream in(json_path);
        if (!in) throw std::runtime_error("Could not open config file: " + json_path);

        json j;
        try {
            in >> j;
        }
        catch (const std::exception& e) {
            throw std::runtime_error(std::string("Invalid JSON in config file: ") + e.what());
        }

        FilterConfig cfg;

        // Scalars (numbers)
        set_if_number(j, "expression_cutoff", cfg.expression_cutoff);
        set_if_number(j, "min_cells", cfg.min_cells);
        set_if_number(j, "min_genes", cfg.min_genes);
        set_if_number(j, "log2_cutoffh", cfg.log2_cutoffh);
        set_if_number(j, "log2_cutoffl", cfg.log2_cutoffl);

        // --- NEW: Palma params ---
        set_if_number(j, "palma_alpha", cfg.palma_alpha);
        set_if_number(j, "palma_upper", cfg.palma_upper);
        set_if_number(j, "palma_lower", cfg.palma_lower);
        set_if_number(j, "palma_winsor", cfg.palma_winsor);

        // NEW: feature panel sizes
        set_if_number(j, "gini_nfeatures", cfg.gini_nfeatures);
        set_if_number(j, "fano_nfeatures", cfg.fano_nfeatures);
        set_if_number(j, "palma_nfeatures", cfg.palma_nfeatures);

        // Graph mix weights
        set_if_number(j, "gini_balance", cfg.gini_balance);
        set_if_number(j, "fano_balance", cfg.fano_balance);
        set_if_number(j, "palma_balance", cfg.palma_balance);

        // Strings
        set_if(j, "preprocess_method", cfg.preprocess_method);
        set_if(j, "output_folder", cfg.output_folder);
        set_if(j, "gene_name_column", cfg.gene_name_column);
        set_if(j, "cell_type_column", cfg.cell_type_column);

        // Booleans
        set_if(j, "write_dense_csv", cfg.write_dense_csv);
        set_if(j, "write_coo_csv", cfg.write_coo_csv);
        set_if(j, "remove_mir", cfg.remove_mir);

        // Normalize preprocess_method to lowercase
        std::transform(cfg.preprocess_method.begin(), cfg.preprocess_method.end(),
            cfg.preprocess_method.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        // Guard: only "cpm" or "none"
        if (cfg.preprocess_method != "cpm" && cfg.preprocess_method != "none") {
            throw std::runtime_error("preprocess_method must be \"cpm\" or \"none\"; got: " + cfg.preprocess_method);
        }

        return cfg;
    }

} // namespace rarecell
