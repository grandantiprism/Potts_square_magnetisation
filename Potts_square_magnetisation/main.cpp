#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <ctime> // 時間取得用

using namespace std;
namespace fs = std::filesystem;

// --- パラメータ設定 ---
const int Q = 4;             // Potts状態数
const int L = 128;            // 格子サイズ
const int N = L * L;
const int MCS = 1000000;     // 本測定のステップ数
const int THERM = L * 20;    // 熱平衡化ステップ数

const double beta_min = 1.095; // q=3 0.995
const double beta_max = 1.105; // q=3 1.015
const int num_beta = 20;

struct Potts2D {
    int L;
    int q;
    vector<int> spins;
    mt19937 gen;
    uniform_int_distribution<int> dist_site;
    uniform_int_distribution<int> dist_q;
    uniform_real_distribution<double> dist_prob;

    Potts2D(int l, int q_val, int seed) : L(l), q(q_val), spins(l * l), gen(seed),
                                          dist_site(0, l * l - 1),
                                          dist_q(1, q_val),
                                          dist_prob(0.0, 1.0) {
        for (int i = 0; i < L * L; ++i) spins[i] = dist_q(gen);
    }

    int get_idx(int x, int y) {
        return ((x + L) % L) * L + ((y + L) % L);
    }

    double calc_magnetization() {
        vector<int> counts(q + 1, 0);
        for (int s : spins) counts[s]++;
        int n_max = *max_element(counts.begin() + 1, counts.end());
        return (static_cast<double>(q) * n_max / (L * L) - 1.0) / (q - 1.0);
    }

    void wolff_step(double beta) {
        double p_add = 1.0 - exp(-beta);
        int root = dist_site(gen);
        int old_spin = spins[root];
        
        int new_spin;
        do { new_spin = dist_q(gen); } while (new_spin == old_spin);

        vector<int> cluster_stack;
        cluster_stack.push_back(root);
        spins[root] = new_spin;

        while (!cluster_stack.empty()) {
            int curr = cluster_stack.back();
            cluster_stack.pop_back();

            int x = curr / L;
            int y = curr % L;

            int neighbors[4] = { get_idx(x+1, y), get_idx(x-1, y), get_idx(x, y+1), get_idx(x, y-1) };

            for (int next : neighbors) {
                if (spins[next] == old_spin) {
                    if (dist_prob(gen) < p_add) {
                        spins[next] = new_spin;
                        cluster_stack.push_back(next);
                    }
                }
            }
        }
    }
};

int main() {
    // 出力ディレクトリの作成
    string dir_name = "output_potts/q" + to_string(Q) + "_" + to_string(L) + "x" + to_string(L);
    if (!fs::exists(dir_name)) fs::create_directories(dir_name);
    
    auto start = chrono::high_resolution_clock::now();
    double beta_step = (beta_max - beta_min) / num_beta;

    // --- シミュレーションループ ---
    for (int i = 0; i <= num_beta; ++i) {
        double beta = beta_min + i * beta_step;
        Potts2D model(L, Q, 12345); // シード固定

        stringstream ss;
        ss << dir_name << "/beta_" << fixed << setprecision(5) << beta << ".txt";
        ofstream ofs(ss.str(), ios::app);
        ofs << fixed << setprecision(15);
        
        cout << "Simulating q=" << Q << ", beta = " << beta << "..." << endl;

        for (int t = 0; t < THERM; ++t) model.wolff_step(beta);

        for (int t = 0; t < MCS; ++t) {
            model.wolff_step(beta);
            double m = model.calc_magnetization();
            ofs << m << "\n";
        }
        ofs.close();
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // --- ログファイルへの書き出し ---
    ofstream log_file("log_potts.txt", ios::app);
    if (log_file) {
        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        
        log_file << "--- Potts Model Simulation Log ---" << endl;
        log_file << "Date: " << ctime(&now);
        log_file << "Q (States): " << Q << endl;
        log_file << "L: " << L << ", MCS: " << MCS << ", THERM: " << THERM << endl;
        log_file << "Beta range: " << beta_min << " to " << beta_max << " (Steps: " << num_beta << ")" << endl;
        log_file << "Elapsed time: " << fixed << setprecision(2) << elapsed.count() << " seconds" << endl;
        log_file << "----------------------------------" << endl << endl;
        log_file.close();
    }

    cout << "Simulation completed in " << elapsed.count() << " seconds." << endl;
    return 0;
}
