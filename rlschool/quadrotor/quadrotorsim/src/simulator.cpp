//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// reference paper:
// Comprehensive Simulation of Quadrotor UAVs Using ROS and Gazebo, Meyer et al.
#include "simulator.h"
#include "common_tools.h"
#include <exception>
#include <iostream>
#include <map>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace quadrotorsim {
Simulator::Simulator() {
    reset();
    _configured = false;
    _precision = 1.0e-4;
}

Simulator::~Simulator() {}

int Simulator::get_config(const char *filename) {
    try {
        boost::property_tree::ptree pt;
        boost::property_tree::read_xml(filename, pt);
        _quality = pt.get<double>("simulator.quality");
        Eigen::Matrix3f inertia;
        inertia(0, 0) = pt.get<double>("simulator.inertia.xx");
        inertia(0, 1) = pt.get<double>("simulator.inertia.xy");
        inertia(0, 2) = pt.get<double>("simulator.inertia.xz");
        inertia(1, 0) = pt.get<double>("simulator.inertia.xy");
        inertia(1, 1) = pt.get<double>("simulator.inertia.yy");
        inertia(1, 2) = pt.get<double>("simulator.inertia.yz");
        inertia(2, 0) = pt.get<double>("simulator.inertia.xz");
        inertia(2, 1) = pt.get<double>("simulator.inertia.yz");
        inertia(2, 2) = pt.get<double>("simulator.inertia.zz");
        _inverse_inertia = inertia.inverse();
        _drag_coeff_momentum = Eigen::Matrix3f::Identity();
        _drag_coeff_force = Eigen::Matrix3f::Identity();
        _drag_coeff_momentum(0, 0) = pt.get<double>("simulator.drag.m_xx");
        _drag_coeff_momentum(1, 1) = pt.get<double>("simulator.drag.m_yy");
        _drag_coeff_momentum(2, 2) = pt.get<double>("simulator.drag.m_zz");
        _drag_coeff_force(0, 0) = pt.get<double>("simulator.drag.f_xx");
        _drag_coeff_force(1, 1) = pt.get<double>("simulator.drag.f_yy");
        _drag_coeff_force(2, 2) = pt.get<double>("simulator.drag.f_zz");
        _gravity_center(0) = pt.get<double>("simulator.gravity_center.x");
        _gravity_center(1) = pt.get<double>("simulator.gravity_center.y");
        _gravity_center(2) = pt.get<double>("simulator.gravity_center.z");
        _thrust_coeff_ct_0 = pt.get<double>("simulator.thrust.CT.0");
        _thrust_coeff_ct_1 = pt.get<double>("simulator.thrust.CT.1");
        _thrust_coeff_ct_2 = pt.get<double>("simulator.thrust.CT.2");
        _thrust_coeff_mm = pt.get<double>("simulator.thrust.Mm");
        _thrust_coeff_jm = pt.get<double>("simulator.thrust.Jm");
        _thrust_coeff_phi = pt.get<double>("simulator.thrust.phi");
        _thrust_coeff_ra = pt.get<double>("simulator.thrust.RA");

        _fail_max_velocity = pt.get<double>("simulator.fail.velocity");
        _fail_max_range = pt.get<double>("simulator.fail.range");
        _fail_max_angular_velocity = pt.get<double>("simulator.fail.w");

        _propeller_coord(0, 0) = pt.get<double>("simulator.propeller.1.x");
        _propeller_coord(0, 1) = pt.get<double>("simulator.propeller.1.y");
        _propeller_coord(0, 2) = pt.get<double>("simulator.propeller.1.z");

        _propeller_coord(1, 0) = pt.get<double>("simulator.propeller.2.x");
        _propeller_coord(1, 1) = pt.get<double>("simulator.propeller.2.y");
        _propeller_coord(1, 2) = pt.get<double>("simulator.propeller.2.z");

        _propeller_coord(2, 0) = pt.get<double>("simulator.propeller.3.x");
        _propeller_coord(2, 1) = pt.get<double>("simulator.propeller.3.y");
        _propeller_coord(2, 2) = pt.get<double>("simulator.propeller.3.z");

        _propeller_coord(3, 0) = pt.get<double>("simulator.propeller.4.x");
        _propeller_coord(3, 1) = pt.get<double>("simulator.propeller.4.y");
        _propeller_coord(3, 2) = pt.get<double>("simulator.propeller.4.z");

        _max_voltage = pt.get<double>("simulator.electric.max_voltage");
        _min_voltage = pt.get<double>("simulator.electric.min_voltage");

        _precision = pt.get<double>("simulator.precision");
        _configured = true;
    } catch (std::exception &error) {
        std::string msg("Error in loading configuration: ");
        msg = msg + error.what();
        throw std::runtime_error(msg);
        return -1;
    }
    return 0;
}

py::dict Simulator::py_get_config(py::str filename) {
    std::string file = py::cast<std::string>(filename);
    get_config(file.c_str());

    using namespace pybind11::literals;
    return py::dict("action_space_low"_a = _min_voltage,
                    "action_space_high"_a = _max_voltage,
                    "range"_a = _fail_max_range);
}

void Simulator::reset() {
    _state = ZERO_STATE;
    _body_acceleration = Eigen::Vector3f(0.0, 0.0, 0.0);
    _coordinate_converter_to_world = _state.rotation_matrix;
    _coordinate_converter_to_body = _state.rotation_matrix.inverse();
}

void Simulator::reset(State &state) {
    _state = state;
    _body_acceleration = Eigen::Vector3f(0.0, 0.0, 0.0);
    _coordinate_converter_to_world = _state.rotation_matrix;
    _coordinate_converter_to_body = _state.rotation_matrix.inverse();
}

void Simulator::reset(double g_x, double g_y, double g_z, double g_v_x,
                      double g_v_y, double g_v_z, double w_x, double w_y,
                      double w_z, double roll, double pitch, double yaw) {
    _state.global_position(0) = g_x;
    _state.global_position(1) = g_y;
    _state.global_position(2) = g_z;
    _state.global_velocity(0) = g_v_x;
    _state.global_velocity(1) = g_v_y;
    _state.global_velocity(2) = g_v_z;
    _state.body_angular_velocity(0) = w_x;
    _state.body_angular_velocity(1) = w_y;
    _state.body_angular_velocity(2) = w_z;
    _state.propeller_angular_velocity(0) = 0;
    _state.propeller_angular_velocity(1) = 0;
    _state.propeller_angular_velocity(2) = 0;
    attitude_to_rotation(roll, pitch, yaw, _state.rotation_matrix);
    _body_acceleration = Eigen::Vector3f(0.0, 0.0, 0.0);
    _coordinate_converter_to_world = _state.rotation_matrix;
    _coordinate_converter_to_body = _state.rotation_matrix.inverse();
}

void Simulator::py_reset(py::kwargs kwargs) {
    std::map<std::string, double> kwargs_map;
    std::string kwlist[] = {"roll",  "pitch", "yaw",   "w_x", "w_y", "w_z",
                            "g_v_x", "g_v_y", "g_v_z", "x",   "y",   "z"};

    for (std::string kw : kwlist) {
        if (kwargs.contains(kw))
            kwargs_map[kw] = py::cast<double>(kwargs[kw.c_str()]);
        else
            kwargs_map[kw] = 0.0;
    }

    reset(kwargs_map["x"], kwargs_map["y"], kwargs_map["z"],
          kwargs_map["g_v_x"], kwargs_map["g_v_y"], kwargs_map["g_v_z"],
          kwargs_map["w_x"], kwargs_map["w_y"], kwargs_map["w_z"],
          kwargs_map["roll"], kwargs_map["pitch"], kwargs_map["yaw"]);
}

void Simulator::get_printable_state(PrintableState &state) {
    state.g_x = _state.global_position(0);
    state.g_y = _state.global_position(1);
    state.g_z = _state.global_position(2);
    state.g_v_x = _state.global_velocity(0);
    state.g_v_y = _state.global_velocity(1);
    state.g_v_z = _state.global_velocity(2);
    Eigen::Vector3f body_velocity =
        _coordinate_converter_to_body * _state.global_velocity;
    state.b_v_x = body_velocity(0);
    state.b_v_y = body_velocity(1);
    state.b_v_z = body_velocity(2);
    state.w_x = _state.body_angular_velocity(0);
    state.w_y = _state.body_angular_velocity(1);
    state.w_z = _state.body_angular_velocity(2);
    rotation_to_attitude(_state.rotation_matrix, state.roll, state.pitch,
                         state.yaw);
    state.power = _state.power;
}

py::dict Simulator::py_get_state() {
    PrintableState state;
    get_printable_state(state);

    using namespace pybind11::literals;
    return py::dict(
        "x"_a = state.g_x, "y"_a = state.g_y, "z"_a = state.g_z,
        "g_v_x"_a = state.g_v_x, "g_v_y"_a = state.g_v_y,
        "g_v_z"_a = state.g_v_z, "b_v_x"_a = state.b_v_x,
        "b_v_y"_a = state.b_v_y, "b_v_z"_a = state.b_v_z, "w_x"_a = state.w_x,
        "w_y"_a = state.w_y, "w_z"_a = state.w_z, "roll"_a = state.roll,
        "pitch"_a = state.pitch, "yaw"_a = state.yaw, "power"_a = state.power);
}

// Read output from sensor
// To be updated: noises not added
void Simulator::read_sensor(SensorOutput &output) {
    // std::cout << "Coord to body:\n"
    //           << _coordinate_converter_to_body << std::endl;
    // std::cout << "Body acc:\n" << _body_acceleration << std::endl;
    Eigen::Vector3f body_velocity =
        _coordinate_converter_to_body * _state.global_velocity;
    output.imu_acc = _body_acceleration +
                     _coordinate_converter_to_body *
                         Eigen::Vector3f(0.0, 0.0, GRAVITY_ACCELERATION);
    output.imu_gyro = _state.body_angular_velocity;
    output.vio = _coordinate_converter_to_body * _state.global_velocity;
}

py::dict Simulator::py_get_sensor() {
    SensorOutput output;
    read_sensor(output);

    using namespace pybind11::literals;
    return py::dict(
        "acc_x"_a = output.imu_acc(0), "acc_y"_a = output.imu_acc(1),
        "acc_z"_a = output.imu_acc(2), "gyro_x"_a = output.imu_gyro(0),
        "gyro_y"_a = output.imu_gyro(1), "gyro_z"_a = output.imu_gyro(2));
}

int Simulator::run_time(ActionU &act, double dt) {
    unsigned int times = 0;
    if (not _configured) {
        std::cerr << "The simulator is not configured" << std::endl;
        return -1;
    }

    if (_precision < 1.0e-8 or _precision > dt) {
        std::cerr << "Inproper parameter, quit" << std::endl;
        return -1;
    }

    times = int(dt / _precision);
    for (unsigned int i = 0; i < times; i++) {
        if (run_internal(act) != 0) {
            return -1;
        }
    }
    return 0;
}

void Simulator::py_step(py::list act, float dt) {
    if (act.size() != 4)
        throw std::invalid_argument(
            "Action should be a list with 4 float numbers.");

    float u1 = act[0].cast<float>();
    float u2 = act[1].cast<float>();
    float u3 = act[2].cast<float>();
    float u4 = act[3].cast<float>();
    ActionU actU(u1, u2, u3, u4);
    int ret = run_time(actU, dt);

    if (ret != 0)
        throw std::runtime_error(
            "Unexpected Error In simulation, Check if correctly configured");
}

int Simulator::check_failure() {
    if (_state.global_position.norm() > _fail_max_range) {
        std::cerr << "The UAV exits the valid zone" << std::endl;
        return -1;
    }
    if (_state.body_angular_velocity.norm() > _fail_max_angular_velocity) {
        std::cerr << "The UAV has too large angualr velocity to recover"
                  << std::endl;
        return -1;
    }
    if (_state.global_velocity.norm() > _fail_max_velocity) {
        std::cerr << "The UAV has too large velocity to recover" << std::endl;
        return -1;
    }
    return 0;
}

py::list Simulator::define_velocity_control_task(double dt, int nt, int seed) {
    State current_state;
    Eigen::Vector3f body_acc;
    save_state(current_state, body_acc);

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(_min_voltage,
                                                        _max_voltage);

    py::list velocity_list;
    for (int i = 0; i < nt; i++) {
        double act[4];
        for (int j = 0; j < 4; j++)
            act[j] = distribution(generator);

        ActionU actU(act[0], act[1], act[2], act[3]);
        int ret = run_time(actU, dt);

        if (ret != 0)
            throw std::runtime_error("Unexpected Error In simulation, Check if "
                                     "correctly configured");
        py::tuple velocity =
            py::make_tuple(_state.global_velocity(0), _state.global_velocity(1),
                           _state.global_velocity(2));
        velocity_list.append(velocity);
    }

    restore_state(current_state, body_acc);

    return velocity_list;
}

int Simulator::run_internal(ActionU &act) {
    if (not _configured) {
        return -1;
    }
    // update propellar state
    Eigen::Vector3f prop_force(0.0, 0.0, 0.0);
    Eigen::Vector3f prop_torque(0.0, 0.0, 0.0);
    std::vector<double> me;
    me.resize(4);
    std::vector<double> propeller_powers;
    propeller_powers.resize(4);
    std::vector<double> w_m_vec;
    w_m_vec.resize(4);
    for (int i = 0; i < 4; i++) {
        double eff_act = act(i);
        if (act(i) > _max_voltage) {
            eff_act = _max_voltage;
        } else if (act(i) < _min_voltage) {
            eff_act = _min_voltage;
        }
        // NOTE: act is voltages U of four propellers
        // me[i] = \phi * I_A is equation (4) in paper
        // use me[i] / \phi * U to get power of i-th propeller
        me[i] = _thrust_coeff_phi / _thrust_coeff_ra *
                (eff_act -
                 _thrust_coeff_phi * _state.propeller_angular_velocity(i));
        if (me[i] < 0)
            propeller_powers[i] = -me[i] / _thrust_coeff_phi * eff_act;
        else
            propeller_powers[i] = me[i] / _thrust_coeff_phi * eff_act;
        // std::cout << "phi: " << _thrust_coeff_phi << ", " << "ra: " << _thrust_coeff_ra << std::endl;
        // std::cout << "w_" << i << ": " << _state.propeller_angular_velocity(i) << std::endl;
        // std::cout << "me_" << i << ": " << me[i] << std::endl;
        double d_propeller_angular_velocity =
            1.0 / _thrust_coeff_jm * (me[i] - _thrust_coeff_mm);

        // update the propellar angular velocity
        double w_m = _state.propeller_angular_velocity(i) +
                     _precision * d_propeller_angular_velocity;
        double l_m = _propeller_coord.row(i).norm();
        Eigen::Vector3f body_velocity =
            _coordinate_converter_to_body * _state.global_velocity;
        double v_1 =
            body_velocity(2) +
            (_state.body_angular_velocity.cross(_propeller_coord.row(i)) *
             l_m)(2);
        double sign = (v_1 > 0.0) ? 1.0 : -1.0;

        // calculate thrust
        double thrust = _thrust_coeff_ct_0 * w_m * w_m +
                        _thrust_coeff_ct_1 * w_m * v_1 +
                        _thrust_coeff_ct_2 * v_1 * v_1 * sign;
        double w_m_when_thrust_lowest =
            -_thrust_coeff_ct_1 * v_1 / (2 * _thrust_coeff_ct_0);
        w_m_vec[i] = w_m_when_thrust_lowest;
        _state.propeller_angular_velocity(i) = w_m;

        prop_force(2) += thrust;
        prop_torque +=
            -Eigen::Vector3f(0.0, 0.0, thrust).cross(_propeller_coord.row(i));
    }
    prop_torque(2) += -me[0] + me[1] - me[2] + me[3];
    // std::cout << "prop_force: [" << prop_force(0) << ", " << prop_force(1) << ", " << prop_force(2) << "]" << std::endl;
    // std::cout << "prop_torque: [" << prop_torque(0) << ", " << prop_torque(1) << ", " << prop_torque(2) << "]" << std::endl;

    // drag force
    Eigen::Vector3f f_drag = -_state.global_velocity.norm() *
                             _drag_coeff_force * _coordinate_converter_to_body *
                             _state.global_velocity;

    // drag torque
    Eigen::Vector3f t_drag = -_state.body_angular_velocity.norm() *
                             _drag_coeff_momentum *
                             _state.body_angular_velocity;
    // gravity force
    Eigen::Vector3f grav(0.0, 0.0, -GRAVITY_ACCELERATION);
    Eigen::Vector3f f_grav = _coordinate_converter_to_body * grav * _quality;
    // gravity torque
    Eigen::Vector3f t_grav = -f_grav.cross(_gravity_center);

    // calculate summation forces and torques
    Eigen::Vector3f f_all = prop_force + f_grav + f_drag;
    Eigen::Vector3f t_all = prop_torque + t_grav + t_drag;
    // std::cout << "f_all: [" << f_all(0) << ", " << f_all(1) << ", " << f_all(2) << "]" << std::endl;
    // std::cout << "t_all: [" << t_all(0) << ", " << t_all(1) << ", " << t_all(2) << "]" << std::endl;

    // update the state
    _body_acceleration = f_all / _quality;
    Eigen::Vector3f acceleration =
        _coordinate_converter_to_world * _body_acceleration;
    _state.global_position += _state.global_velocity * _precision +
                              0.5 * _precision * _precision * acceleration;
    _state.global_velocity += _precision * acceleration;
    _state.power = propeller_powers[0] + propeller_powers[1] +
                   propeller_powers[2] + propeller_powers[3];

    Eigen::Vector3f tmp_angular_velocity =
        _state.body_angular_velocity +
        0.5 * _precision * _inverse_inertia * t_all;

    Eigen::Matrix3f skew_sym_angular_velocity;

    skew_sym_angular_velocity(0, 0) = 0.0;
    skew_sym_angular_velocity(0, 1) = -tmp_angular_velocity(2);
    skew_sym_angular_velocity(0, 2) = tmp_angular_velocity(1);
    skew_sym_angular_velocity(1, 0) = tmp_angular_velocity(2);
    skew_sym_angular_velocity(1, 1) = 0.0;
    skew_sym_angular_velocity(1, 2) = -tmp_angular_velocity(0);
    skew_sym_angular_velocity(2, 0) = -tmp_angular_velocity(1);
    skew_sym_angular_velocity(2, 1) = tmp_angular_velocity(0);
    skew_sym_angular_velocity(2, 2) = 0.0;
    _state.rotation_matrix +=
        _precision * _state.rotation_matrix * skew_sym_angular_velocity;
    _state.body_angular_velocity += _precision * _inverse_inertia * t_all;

    // update coordinate converter
    _coordinate_converter_to_world = _state.rotation_matrix;
    _coordinate_converter_to_body = _state.rotation_matrix.inverse();

    if (check_failure()) {
        std::cerr << "The quadrotor returns a failure";
        return -1;
    }
    return 0;
}

void Simulator::save_state(State &state, Eigen::Vector3f &body_acc) {
    state = _state;
    body_acc = _body_acceleration;
}

void Simulator::restore_state(State &state, Eigen::Vector3f &body_acc) {
    _state = state;
    _body_acceleration = body_acc;
    _coordinate_converter_to_world = _state.rotation_matrix;
    _coordinate_converter_to_body = _state.rotation_matrix.inverse();
}

}; // namespace quadrotorsim

PYBIND11_MODULE(quadrotorsim, m) {
    m.doc() = "A Quadrotor Simulator";

    using namespace quadrotorsim;
    py::class_<Simulator>(m, "Simulator")
        .def(py::init())
        .def("get_config", &Simulator::py_get_config)
        .def("reset", &Simulator::py_reset)
        .def("get_state", &Simulator::py_get_state)
        .def("get_sensor", &Simulator::py_get_sensor)
        .def("step", &Simulator::py_step)
        .def("define_velocity_control_task",
             &Simulator::define_velocity_control_task);
}
