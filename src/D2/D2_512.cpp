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
#include <limits>
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
    vector<float> lo;
    vector<float> hi;
    vector<Node*> children;

    bool isleaf() const override { return false; }

    void add(const MBR& mbr, Node* child) {
        lo.push_back(mbr.min_x);
        lo.push_back(mbr.min_y);
        hi.push_back(mbr.max_x);
        hi.push_back(mbr.max_y);
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

MBR calculateMBRFromD2Vectors(const vector<float>& lo,
                               const vector<float>& hi,
                               int id) {
    MBR mbr;
    mbr.id = id;

    mbr.min_x = mbr.min_y = numeric_limits<float>::max();
    mbr.max_x = mbr.max_y = numeric_limits<float>::lowest();

    for (size_t i = 0; i < lo.size(); i += 2) {
        mbr.min_x = min(mbr.min_x, lo[i]);
        mbr.min_y = min(mbr.min_y, lo[i + 1]);
    }

    for (size_t i = 0; i < hi.size(); i += 2) {
        mbr.max_x = max(mbr.max_x, hi[i]);
        mbr.max_y = max(mbr.max_y, hi[i + 1]);
    }

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

        MBR parent_mbr = calculateMBRFromD2Vectors(
            internal_node->lo, internal_node->hi, internal_node->id
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

            for (size_t k = j; k < sub_end; ++k) {
                leaf->add(points[k]);
            }

            MBR mbr = calculateMBR(leaf->points, leaf->id);
            created_leaves.push_back({static_cast<Node*>(leaf), mbr});
        }
    }

    createNextLevel(created_leaves);
}

void writeRtreeToFile(const string& filename) {
    ofstream outFile(filename);
    if (!outFile) {
        cerr << "Error creating output file: " << filename << endl;
        return;
    }

    outFile << fixed << setprecision(15);

    outFile << "Leaf Nodes\n";
    for (const auto& node : created_leaves) {
        LeafNode* leaf = static_cast<LeafNode*>(node.first);
        outFile << "Leaf ID: " << leaf->id << "\n";
        for (const auto& point : leaf->points) {
            outFile << "  Coordinate ID: " << point.id
                    << ", X: " << point.x
                    << ", Y: " << point.y << "\n";
        }
    }

    outFile << "\nInternal Nodes\n";
    for (const auto& node : internal_nodes) {
        InternalNode* inner = static_cast<InternalNode*>(node.first);
        outFile << "Internal Node ID: " << inner->id << "\n";

        outFile << "  LowXY:";
        for (size_t i = 0; i < inner->lo.size(); i += 2) {
            outFile << " (" << inner->lo[i] << ", " << inner->lo[i + 1] << ")";
        }
        outFile << "\n";

        outFile << "  HighXY:";
        for (size_t i = 0; i < inner->hi.size(); i += 2) {
            outFile << " (" << inner->hi[i] << ", " << inner->hi[i + 1] << ")";
        }
        outFile << "\n";

        outFile << "  Pointers:";
        for (Node* child : inner->children)
            outFile << " " << child->id;
        outFile << "\n\n";
    }

    outFile.close();
    cout << "R-tree written to " << filename << endl;
}

int spatialSelectD2(const Query& query) {
    int count = 0;
    queue<Node*> q;
    if (root) {
        q.push(root);
    }

    __m512 query_mins = _mm512_set_ps(
        query.min_y, query.min_x, query.min_y, query.min_x,
        query.min_y, query.min_x, query.min_y, query.min_x,
        query.min_y, query.min_x, query.min_y, query.min_x,
        query.min_y, query.min_x, query.min_y, query.min_x
    );
    __m512 query_maxs = _mm512_set_ps(
        query.max_y, query.max_x, query.max_y, query.max_x,
        query.max_y, query.max_x, query.max_y, query.max_x,
        query.max_y, query.max_x, query.max_y, query.max_x,
        query.max_y, query.max_x, query.max_y, query.max_x
    );

    while (!q.empty()) {
        Node* current = q.front(); q.pop();

        if (current->isleaf()) {
            LeafNode* leaf = static_cast<LeafNode*>(current);
            for (const auto& point : leaf->points) {
                if (point.x >= query.min_x && point.x <= query.max_x &&
                    point.y >= query.min_y && point.y <= query.max_y) {
                    count++;
                }
            }
        } else {
            InternalNode* inner = static_cast<InternalNode*>(current);
            const int n = inner->children.size();

            for (int i = 0; i < n; i += 8) {
                if (i * 2 + 15 >= inner->lo.size()) {
                    for (int j = i; j < n; ++j) {
                        float min_x = inner->lo[2 * j];
                        float min_y = inner->lo[2 * j + 1];
                        float max_x = inner->hi[2 * j];
                        float max_y = inner->hi[2 * j + 1];
                        if (max_x >= query.min_x && min_x <= query.max_x &&
                            max_y >= query.min_y && min_y <= query.max_y) {
                            q.push(inner->children[j]);
                        }
                    }
                    break;
                }

                __m512 lo_vec = _mm512_loadu_ps(&inner->lo[2 * i]);
                __m512 hi_vec = _mm512_loadu_ps(&inner->hi[2 * i]);

                __mmask16 cmp1 = _mm512_cmp_ps_mask(hi_vec, query_mins, _CMP_GE_OQ);
                __mmask16 cmp2 = _mm512_cmp_ps_mask(lo_vec, query_maxs, _CMP_LE_OQ);

                __mmask16 bitmask = cmp1 & cmp2;
                
                if ((bitmask & 0x3) == 0x3) {
                    q.push(inner->children[i]);
                }
                if ((bitmask & 0xC) == 0xC) {
                    q.push(inner->children[i + 1]);
                }
                if ((bitmask & 0x30) == 0x30) {
                    q.push(inner->children[i + 2]);
                }
                if ((bitmask & 0xC0) == 0xC0) {
                    q.push(inner->children[i + 3]);
                }
                if ((bitmask & 0x300) == 0x300) {
                    q.push(inner->children[i + 4]);
                }
                if ((bitmask & 0xC00) == 0xC00) {
                    q.push(inner->children[i + 5]);
                }
                if ((bitmask & 0x3000) == 0x3000) {
                    q.push(inner->children[i + 6]);
                }
                if ((bitmask & 0xC000) == 0xC000) {
                    q.push(inner->children[i + 7]);
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
        matched_points = spatialSelectD2(q);
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
