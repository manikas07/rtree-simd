#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <tuple>
#include <sstream>
#include <unordered_set>
#include <queue>
#include <chrono>
#include <immintrin.h>
using namespace std::chrono;
using namespace std;

int fanout;
int node_id_counter = 0;

struct Point {
    int id;
    float x, y;
};

struct Query {
    float min_x, min_y, max_x, max_y;
};

struct MBR {
    int id;
    float min_x, min_y, max_x, max_y;
};

class Node {
public:
    virtual ~Node() = default;
    virtual bool isleaf() const = 0;
    int id;
};

class LeafNode : public Node {
public:
    vector<Point> points;
    bool isleaf() const override { return true; }
    void add(const Point& point) {
        points.push_back(point);
    }
};

class InternalNode : public Node {
public:
    vector<float> min_xs, min_ys, max_xs, max_ys;
    vector<Node*> children;

    bool isleaf() const override { return false; }

    void add(const MBR& mbr, Node* child) {
        min_xs.push_back(mbr.min_x);
        min_ys.push_back(mbr.min_y);
        max_xs.push_back(mbr.max_x);
        max_ys.push_back(mbr.max_y);
        children.push_back(child);
    }
};

vector<pair<Node*, MBR>> created_leaves;
vector<pair<Node*, MBR>> internal_nodes;
vector<Point> input_points;
Node* root = nullptr;
vector<Query> queries;

bool compareByX(const Point& a, const Point& b) { 
    return a.x < b.x; 
}
bool compareByY(const Point& a, const Point& b) { 
    return a.y < b.y; 
}

MBR calculateMBR(const vector<Point>& points, int id) {
    MBR mbr;
    mbr.id = id;
    mbr.min_x = mbr.max_x = points[0].x;
    mbr.min_y = mbr.max_y = points[0].y;
    for (const auto& point : points) {
        mbr.min_x = min(mbr.min_x, point.x);
        mbr.max_x = max(mbr.max_x, point.x);
        mbr.min_y = min(mbr.min_y, point.y);
        mbr.max_y = max(mbr.max_y, point.y);
    }
    return mbr;
}

MBR calculateMBRFromVectors(const vector<float>& min_xs,
                             const vector<float>& min_ys,
                             const vector<float>& max_xs,
                             const vector<float>& max_ys,
                             int id) {
    MBR mbr;
    mbr.id = id;
    mbr.min_x = *min_element(min_xs.begin(), min_xs.end());
    mbr.min_y = *min_element(min_ys.begin(), min_ys.end());
    mbr.max_x = *max_element(max_xs.begin(), max_xs.end());
    mbr.max_y = *max_element(max_ys.begin(), max_ys.end());
    return mbr;
}

void createNextLevel(vector<pair<Node*, MBR>> input_nodes) {
    vector<pair<Node*, MBR>> next_level_nodes;
    size_t count = input_nodes.size();
    for (size_t i = 0; i < count; i += fanout) {
        size_t end = min(i + fanout, count);
        InternalNode* internal_node = new InternalNode();
        internal_node->id = node_id_counter++;
        for (int j = i; j < end; ++j) {
            internal_node->add(input_nodes[j].second, input_nodes[j].first);
        }
        MBR parent_mbr = calculateMBRFromVectors(
            internal_node->min_xs, 
            internal_node->min_ys,
            internal_node->max_xs, 
            internal_node->max_ys,
            internal_node->id
        );
        internal_nodes.push_back({static_cast<Node*>(internal_node), parent_mbr});
        next_level_nodes.push_back({static_cast<Node*>(internal_node), parent_mbr});
    }
    if (next_level_nodes.size() > 1) {
        createNextLevel(next_level_nodes);
    } else {
        root = next_level_nodes[0].first;
    }
}

void createRtree(vector<Point>& points) {
    sort(points.begin(), points.end(), compareByX);
    int num_leafs = ceil(points.size() / static_cast<double>(fanout));
    int block_size = ceil(sqrt(num_leafs)) * fanout;
    for (size_t i = 0; i < points.size(); i += block_size) {
        size_t end = min(i + block_size, points.size());
        sort(points.begin() + i, points.begin() + end, compareByY);
        for (size_t j = i; j < end; j += fanout) {
            size_t sub_end = min(j + fanout, end);
            LeafNode* leaf = new LeafNode();
            leaf->id = node_id_counter++;
            for (size_t k = j; k < sub_end; ++k)
                leaf->add(points[k]);
            MBR mbr = calculateMBR(leaf->points, leaf->id);
            created_leaves.push_back({static_cast<Node*>(leaf), mbr});
        }
    }
    createNextLevel(created_leaves);
}

int spatialSelectD1(const Query& query) {
    int count = 0;
    queue<Node*> q;
    q.push(root);
    __m256 q_min_x = _mm256_set1_ps(query.min_x);
    __m256 q_min_y = _mm256_set1_ps(query.min_y);
    __m256 q_max_x = _mm256_set1_ps(query.max_x);
    __m256 q_max_y = _mm256_set1_ps(query.max_y);

    while (!q.empty()) {
        Node* current = q.front(); q.pop();

        if (current->isleaf()) {
            LeafNode* leaf = static_cast<LeafNode*>(current);
            int m = leaf->points.size();
            for (int i = 0; i < m; i += 8) {
                int len = min(8, m - i);
                float x_vals[8] = {0}, y_vals[8] = {0};
                for (int j = 0; j < len; ++j) {
                    x_vals[j] = leaf->points[i + j].x;
                    y_vals[j] = leaf->points[i + j].y;
                }
                __m256 xs = _mm256_loadu_ps(x_vals);
                __m256 ys = _mm256_loadu_ps(y_vals);
                __m256 cmp_x_min = _mm256_cmp_ps(xs, q_min_x, _CMP_GE_OQ);
                __m256 cmp_x_max = _mm256_cmp_ps(xs, q_max_x, _CMP_LE_OQ);
                __m256 cmp_y_min = _mm256_cmp_ps(ys, q_min_y, _CMP_GE_OQ);
                __m256 cmp_y_max = _mm256_cmp_ps(ys, q_max_y, _CMP_LE_OQ);
                __m256 mask1 = _mm256_and_ps(cmp_x_min, cmp_x_max);
                __m256 mask2 = _mm256_and_ps(cmp_y_min, cmp_y_max);
                __m256 mask = _mm256_and_ps(mask1, mask2);
                int bitmask = _mm256_movemask_ps(mask);
                for (int j = 0; j < len; ++j) {
                    if (bitmask & (1 << j)) {
                        count++;
                    }
                }
            }
        }  else {
            InternalNode* inner = static_cast<InternalNode*>(current);
            int n = inner->min_xs.size();

            if (n < 8) {
                for (int i = 0; i < n; ++i) {
                    if (inner->max_xs[i] >= query.min_x &&
                        inner->min_xs[i] <= query.max_x &&
                        inner->max_ys[i] >= query.min_y &&
                        inner->min_ys[i] <= query.max_y) {
                        q.push(inner->children[i]);
                    }
                }
            } else {
                for (int i = 0; i < n; i += 8) {
                    int len = min(8, n - i);
                    if (len < 8) {  // last partial group: scalar check, do not load past the arrays
                        for (int j = i; j < n; ++j) {
                            if (inner->max_xs[j] >= query.min_x && inner->min_xs[j] <= query.max_x &&
                                inner->max_ys[j] >= query.min_y && inner->min_ys[j] <= query.max_y) {
                                q.push(inner->children[j]);
                            }
                        }
                        break;
                    }

                    __m256 min_x = _mm256_loadu_ps(&inner->min_xs[i]);
                    __m256 min_y = _mm256_loadu_ps(&inner->min_ys[i]);
                    __m256 max_x = _mm256_loadu_ps(&inner->max_xs[i]);
                    __m256 max_y = _mm256_loadu_ps(&inner->max_ys[i]);

                    __m256 cmp1 = _mm256_cmp_ps(max_x, q_min_x, _CMP_GE_OQ);
                    __m256 cmp2 = _mm256_cmp_ps(min_x, q_max_x, _CMP_LE_OQ);
                    __m256 cmp3 = _mm256_cmp_ps(max_y, q_min_y, _CMP_GE_OQ);
                    __m256 cmp4 = _mm256_cmp_ps(min_y, q_max_y, _CMP_LE_OQ);

                    __m256 mask = _mm256_and_ps(_mm256_and_ps(cmp1, cmp2), _mm256_and_ps(cmp3, cmp4));
                    int bitmask = _mm256_movemask_ps(mask);

                    for (int j = 0; j < len; ++j) {
                        if (bitmask & (1 << j)) {
                            q.push(inner->children[i + j]);
                        }
                    }
                }
            }
        }
    }
    return count;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: ./rtree <fanout> <input_file> <output_file> <query_file>\n";
        return 1;
    }

    ifstream file(argv[2]);
    if (!file) {
        cerr << "Error opening input file.\n";
        return 1;
    }

    fanout = atoi(argv[1]);
    string line;
    int id = 0;
    while (getline(file, line)) {
        istringstream iss(line);
        float x, y;
        char comma;
        if (iss >> x >> comma >> y && comma == ',') {
            input_points.push_back({id++, x, y});
        } else {
            cerr << "Invalid line: " << line << endl;
        }
    }
    file.close();

    //cout << "Loaded " << input_points.size() << " points." << endl;

    createRtree(input_points);
    string output_path_rtree = argv[3];
    //writeRtreeToFile(output_path_rtree);

    ifstream inputQueryFile(argv[4]);
    if (!inputQueryFile) {
        cerr << "Error opening file query file." << endl;
        return 1;
    }

    while (getline(inputQueryFile, line)) {
        double min_x, min_y, max_x, max_y;
        char comma;
        stringstream ss(line);
        if (ss >> min_x >> min_y >> comma >> max_x >> max_y && comma == ',') {
            queries.push_back({static_cast<float>(min_x), static_cast<float>(min_y), static_cast<float>(max_x), static_cast<float>(max_y)});    
        } else {
            cerr << "Invalid query format: " << line << endl;
        }
    }
    

    unsigned long long total_time = 0;     // nanoseconds
    unsigned long long total_matched = 0;
    for (int i = 0; i < queries.size(); i++)
    {
        int matched_points = 0;
        auto& q = queries[i];
        
        auto start = high_resolution_clock::now();
        matched_points = spatialSelectD1(q);
        auto end = high_resolution_clock::now();
        
        auto elapsed = duration_cast<nanoseconds>(end - start).count();
        total_time += elapsed;
        total_matched += matched_points;
        
        //cout << "Matched Points: " << matched_points << endl;
        //cout << "Time for Query " << i + 1 << ": " << elapsed / 1000.0 << " ms" << endl;
    }
    cout << "Fanout: " << fanout << endl;
    cout << "Input File: " << argv[2] << endl;
    cout << "Query File: " << argv[4] << endl;
    cout << "Total Elapsed Time: " << total_time / 1e6 << " ms\n";
    cout << "Total Matched Points: " << total_matched << endl;
    cout << endl;
    return 0;
}
