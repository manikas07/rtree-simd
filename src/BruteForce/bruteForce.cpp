#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;

struct Coordinate {
    double x, y;
};

struct Query {
    double min_x, min_y, max_x, max_y;
};

vector<Coordinate> readPoints(const string& filename) {
    ifstream inFile(filename);
    vector<Coordinate> points;
    string line;

    while (getline(inFile, line)) {
        replace(line.begin(), line.end(), ',', ' ');
        stringstream ss(line);
        double x, y;
        if (ss >> x >> y) {
            points.push_back({x, y});
        }
    }

    return points;
}

vector<Query> readQueries(const string& filename) {
    ifstream inFile(filename);
    vector<Query> queries;
    string line;

    while (getline(inFile, line)) {
        replace(line.begin(), line.end(), ',', ' ');
        stringstream ss(line);
        double min_x, min_y, max_x, max_y;
        if (ss >> min_x >> min_y >> max_x >> max_y) {
            queries.push_back({min_x, min_y, max_x, max_y});
        }
    }

    return queries;
}

void bruteForceSearch(const vector<Coordinate>& points, const vector<Query>& queries) {
    unsigned long long total = 0;
    for (size_t i = 0; i < queries.size(); ++i) {
        const Query& q = queries[i];
        vector<Coordinate> matching;

        for (const auto& point : points) {
            if (point.x >= q.min_x && point.x <= q.max_x &&
                point.y >= q.min_y && point.y <= q.max_y) {
                matching.push_back(point);
            }
        }

        // cout << "Query " << i + 1 << " [" << q.min_x << ", " << q.min_y << "] to [" 
        //      << q.max_x << ", " << q.max_y << "]\n";
        cout << "Matched Points: " << matching.size() << "\n";
        total += matching.size();

        // for (const auto& pt : matching) {
        //     cout << "  (" << pt.x << ", " << pt.y << ")\n";
        // }
        // cout << "--------------------------\n";
    }
    cout << "Total Matched Points: " << total << "\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Error in arguments\n";
        return 1;
    }

    vector<Coordinate> points = readPoints(argv[1]);
    vector<Query> queries = readQueries(argv[2]);

    cout << "Points: " << points.size() << "\n";
    cout << "Queries: " << queries.size() << "\n";

    bruteForceSearch(points, queries);

    return 0;
}
