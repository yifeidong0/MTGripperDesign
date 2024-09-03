// polygon.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

bool point_in_polygon(const std::vector<std::pair<double, double>>& vertices, const std::pair<double, double>& point) {
    bool inside = false;
    double x = point.first, y = point.second;
    int n = vertices.size();

    for (int i = 0, j = n - 1; i < n; j = i++) {
        double xi = vertices[i].first, yi = vertices[i].second;
        double xj = vertices[j].first, yj = vertices[j].second;

        bool intersect = ((yi > y) != (yj > y)) &&
                         (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) {
            inside = !inside;
        }
    }
    return inside;
}

double distance_to_polygon(const std::vector<std::pair<double, double>>& vertices, const std::pair<double, double>& point) {
    double min_dist = std::numeric_limits<double>::max();

    for (const auto& vertex : vertices) {
        double dx = vertex.first - point.first;
        double dy = vertex.second - point.second;
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist < min_dist) {
            min_dist = dist;
        }
    }

    return min_dist;
}

py::array_t<bool> is_point_inside_polygon(py::array_t<double> points, py::array_t<double> vertices, double slack) {
    auto buf_points = points.request();
    auto buf_vertices = vertices.request();

    if (buf_points.ndim != 2 || buf_points.shape[1] != 2)
        throw std::runtime_error("Points array must be of shape [N, 2]");

    if (buf_vertices.ndim != 3 || buf_vertices.shape[1] < 3 || buf_vertices.shape[2] != 2)
        throw std::runtime_error("Vertices array must be of shape [N, M, 2] where M >= 3");

    auto result = py::array_t<bool>(buf_points.shape[0]);
    auto result_buf = result.request();

    bool* result_ptr = static_cast<bool*>(result_buf.ptr);
    double* points_ptr = static_cast<double*>(buf_points.ptr);
    double* vertices_ptr = static_cast<double*>(buf_vertices.ptr);

    for (ssize_t i = 0; i < buf_points.shape[0]; i++) {
        std::pair<double, double> point(points_ptr[i * 2], points_ptr[i * 2 + 1]);

        std::vector<std::pair<double, double>> poly_vertices;
        for (ssize_t j = 0; j < buf_vertices.shape[1]; j++) {
            poly_vertices.emplace_back(vertices_ptr[i * buf_vertices.shape[1] * 2 + j * 2],
                                       vertices_ptr[i * buf_vertices.shape[1] * 2 + j * 2 + 1]);
        }

        if (point_in_polygon(poly_vertices, point) || distance_to_polygon(poly_vertices, point) <= slack) {
            result_ptr[i] = true;
        } else {
            result_ptr[i] = false;
        }
    }

    return result;
}

PYBIND11_MODULE(polygon, m) {
    m.def("is_point_inside_polygon", &is_point_inside_polygon, "Check if points are inside polygons or within slack distance",
          py::arg("points"), py::arg("vertices"), py::arg("slack") = 2.0);
}