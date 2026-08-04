/*
  Week 1 MoJoCo Experiment
*/

#include "mujoco.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDuration = 8.0;           // Total simulation time is 8 seconds
constexpr double kControlFrequency = 50.0;  // Control frequency 50 Hz
constexpr double kInitialAngle = 0.35;      // Initial angle 0.35 rad
constexpr double kTargetAngle = 0.75;       // Target angle 0.75 rad
constexpr double kActionAmplitude = 2.0;    // Control torque amplitude 2 N·m
constexpr double kActionFrequency = 0.5;    // Control oscillation frequency 0.5 Hz

struct Transition {
  double time;      // start time of the current control step
  double q;         // current joint angle
  double qvel;      // current joint angular velocity
  double action;    // torque applied in this control step
  double reward;    // immediate reward after execution
  double next_time; // start time of the next control step (= time + actual_control_period)
  double next_q;    // joint angle of the next control step
  double next_qvel; // joint angular velocity of the next control step
};

double wrap_to_pi(double angle) {
  // The angle of the hinge joint can be continuously accumulated.
  // `std::remainder(x, y)` returns the remainder that is mathematically closest to 0,
  // falling within the interval [-y/2, y/2].
  return std::remainder(angle, 2.0 * kPi);
}

// This is a deliberately simple hand-written policy. Later, an RL or VLA
// policy can replace this function without changing the simulation loop.
double choose_action(double time) {
  return kActionAmplitude * std::sin(2.0 * kPi * kActionFrequency * time);
}

// Reward is evaluated after the transition. Larger is better; zero is the
// maximum possible reward at the target with zero velocity and zero action.
double calculate_reward(double next_q, double next_qvel, double action) {
  const double angle_error = wrap_to_pi(kTargetAngle - next_q);
  return -(angle_error * angle_error +
           0.05 * next_qvel * next_qvel +
           0.001 * action * action);
}

void write_csv(const std::vector<Transition>& transitions,
               const std::filesystem::path& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Could not open " + output_path.string());
  }

  output << "time_s,q_rad,qvel_rad_s,action_torque_nm,reward,"
            "next_time_s,next_q_rad,next_qvel_rad_s\n";
  output << std::setprecision(10);
  for (const Transition& item : transitions) {
    output << item.time << ',' << item.q << ',' << item.qvel << ','
           << item.action << ',' << item.reward << ',' << item.next_time << ','
           << item.next_q << ',' << item.next_qvel << '\n';
  }
}

template <typename ValueFunction>
void draw_panel(std::ofstream& svg,
                const std::vector<Transition>& transitions,
                int top,
                const std::string& label,
                const std::string& color,
                ValueFunction value,
                bool draw_reference = false,
                double reference = 0.0) {
  constexpr int kLeft = 100;
  constexpr int kWidth = 1040;
  constexpr int kHeight = 180;

  double minimum = std::numeric_limits<double>::infinity();
  double maximum = -std::numeric_limits<double>::infinity();
  for (const Transition& item : transitions) {
    minimum = std::min(minimum, value(item));
    maximum = std::max(maximum, value(item));
  }
  if (draw_reference) {
    minimum = std::min(minimum, reference);
    maximum = std::max(maximum, reference);
  }
  if (maximum - minimum < 1e-9) {
    minimum -= 1.0;
    maximum += 1.0;
  } else {
    const double padding = 0.1 * (maximum - minimum);
    minimum -= padding;
    maximum += padding;
  }

  const double start_time = transitions.front().next_time;
  const double end_time = transitions.back().next_time;
  const auto x_position = [&](double time) {
    return kLeft + kWidth * (time - start_time) / (end_time - start_time);
  };
  const auto y_position = [&](double sample) {
    return top + kHeight * (maximum - sample) / (maximum - minimum);
  };

  svg << "<rect x=\"" << kLeft << "\" y=\"" << top
      << "\" width=\"" << kWidth << "\" height=\"" << kHeight
      << "\" fill=\"#151922\" stroke=\"#64748b\"/>\n";

  for (int grid = 1; grid < 5; ++grid) {
    const double x = kLeft + kWidth * grid / 5.0;
    const double y = top + kHeight * grid / 5.0;
    svg << "<line x1=\"" << x << "\" y1=\"" << top << "\" x2=\""
        << x << "\" y2=\"" << top + kHeight
        << "\" stroke=\"#293241\"/>\n";
    svg << "<line x1=\"" << kLeft << "\" y1=\"" << y << "\" x2=\""
        << kLeft + kWidth << "\" y2=\"" << y
        << "\" stroke=\"#293241\"/>\n";
  }

  if (draw_reference && reference >= minimum && reference <= maximum) {
    const double y = y_position(reference);
    svg << "<line x1=\"" << kLeft << "\" y1=\"" << y << "\" x2=\""
        << kLeft + kWidth << "\" y2=\"" << y
        << "\" stroke=\"#fb7185\" stroke-width=\"2\" "
           "stroke-dasharray=\"8 7\"/>\n";
  }

  svg << "<polyline fill=\"none\" stroke=\"" << color
      << "\" stroke-width=\"2.5\" points=\"";
  for (const Transition& item : transitions) {
    svg << x_position(item.next_time) << ',' << y_position(value(item)) << ' ';
  }
  svg << "\"/>\n";

  svg << "<text x=\"88\" y=\"" << top + kHeight / 2
      << "\" text-anchor=\"end\" fill=\"#e2e8f0\" font-size=\"17\">"
      << label
      << "</text>\n";
  svg << std::fixed << std::setprecision(2);
  svg << "<text x=\"92\" y=\"" << top + 14
      << "\" text-anchor=\"end\" fill=\"#94a3b8\" font-size=\"13\">"
      << maximum << "</text>\n";
  svg << "<text x=\"92\" y=\"" << top + kHeight
      << "\" text-anchor=\"end\" fill=\"#94a3b8\" font-size=\"13\">"
      << minimum << "</text>\n";
}

void write_svg(const std::vector<Transition>& transitions,
               const std::filesystem::path& output_path) {
  std::ofstream svg(output_path);
  if (!svg) {
    throw std::runtime_error("Could not open " + output_path.string());
  }

  svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" "
         "height=\"800\" viewBox=\"0 0 1200 800\">\n";
  svg << "<rect width=\"1200\" height=\"800\" fill=\"#0b0f16\"/>\n";
  svg << "<g font-family=\"sans-serif\">\n";
  svg << "<text x=\"600\" y=\"38\" text-anchor=\"middle\" "
         "fill=\"#f8fafc\" font-size=\"25\" font-weight=\"bold\">"
         "Week 1 MuJoCo pendulum</text>\n";
  svg << "<text x=\"600\" y=\"63\" text-anchor=\"middle\" "
         "fill=\"#94a3b8\" font-size=\"15\">"
         "orange: simulated value; dashed red: target angle</text>\n";

  draw_panel(svg, transitions, 90, "q (rad)", "#f59e0b",
             [](const Transition& item) { return item.next_q; }, true,
             kTargetAngle);
  draw_panel(svg, transitions, 310, "qdot (rad/s)", "#38bdf8",
             [](const Transition& item) { return item.next_qvel; });
  draw_panel(svg, transitions, 530, "action (N m)", "#a78bfa",
             [](const Transition& item) { return item.action; });

  svg << "<text x=\"620\" y=\"750\" text-anchor=\"middle\" "
         "fill=\"#e2e8f0\" font-size=\"17\">time (s)</text>\n";
  svg << "</g>\n</svg>\n";
}

}  // namespace

int main() {
  constexpr char kModelPath[] = "assets/pendulum.xml";
  constexpr char kCsvPath[] = "build/pendulum_trajectory.csv";
  constexpr char kPlotPath[] = "build/pendulum_plot.svg";

  char error[1024] = {};

  // Read pendulum.xml and compile it into a physical model.
  mjModel* model = mj_loadXML(kModelPath, nullptr, error, sizeof(error));
  if (model == nullptr) {
    std::cerr << "Failed to load " << kModelPath << ": " << error << '\n';
    return 1;
  }

  // Allocate a "state + work area" for this model.
  mjData* data = mj_makeData(model);
  if (data == nullptr) {
    std::cerr << "Failed to allocate mjData\n";
    mj_deleteModel(model);
    return 1;
  }

  try {
    // mj_name2id(model, type, name): finds the numeric index of an object in the model by name.
    const int joint_id = mj_name2id(model, mjOBJ_JOINT, "hinge");             // mjOBJ_JOINT: find joints
    const int actuator_id = mj_name2id(model, mjOBJ_ACTUATOR, "hinge_motor"); // mjOBJ_ACTUATOR: find the executor
    if (joint_id < 0 || actuator_id < 0) {
      throw std::runtime_error("Expected joint or actuator was not found");
    }

    /* MuJoCo stores the states of all joints in a flat, one-dimensional array:
      data->qpos = [q0, q1, q2, ...]      positions of all joints
      data->qvel = [v0, v1, v2, ...]      velocities of all joints */
    const int qpos_index = model->jnt_qposadr[joint_id];
    const int qvel_index = model->jnt_dofadr[joint_id];

    data->qpos[qpos_index] = kInitialAngle;   // initial swing angle
    data->qvel[qvel_index] = 0.0;             // initially at rest
    mj_forward(model, data);                  // synchronize all derived quantities

    const double control_period = 1.0 / kControlFrequency;

    // Number of physics steps per one control update.
    // Wth a 2ms timestep and 50Hz control, this is exactly 10.
    const int steps_per_action = std::max(1, static_cast<int>(std::lround(control_period / model->opt.timestep)));
    
    // Actual control period, adjusted to be an integer multiple of the physics timestep.
    // This avoids timing drift when control_period doesn't divide evenly by timestep.
    const double actual_control_period = steps_per_action * model->opt.timestep;

    // Total number of control transitions over the entire episode.
    // Each transition spans steps_per_action physics steps.
    const int transition_count = static_cast<int>(std::ceil(kDuration / actual_control_period));

    std::vector<Transition> transitions;
    transitions.reserve(transition_count);

    for (int transition = 0; transition < transition_count; ++transition) {
      // Copy s_t before stepping: mj_step mutates mjData in place.
      const double time = data->time;
      const double q = data->qpos[qpos_index];
      const double qvel = data->qvel[qvel_index];
      const double action = std::clamp(choose_action(time), -3.0, 3.0);
      data->ctrl[actuator_id] = action;   // write the action to the control array

      // The policy acts at 50 Hz while physics is integrated at 500 Hz.
      // The same action is held for all physics steps in this interval.
      for (int step = 0; step < steps_per_action; ++step) {
        mj_step(model, data);
      }

      const double next_q = data->qpos[qpos_index];
      const double next_qvel = data->qvel[qvel_index];
      transitions.push_back({time,
                             q,
                             qvel,
                             action,
                             calculate_reward(next_q, next_qvel, action),
                             data->time,
                             next_q,
                             next_qvel});
    }

    std::filesystem::create_directories("build");
    write_csv(transitions, kCsvPath);
    write_svg(transitions, kPlotPath);

    const Transition& last = transitions.back();
    std::cout << "MuJoCo " << mj_versionString() << '\n'
              << "Physics timestep: " << model->opt.timestep << " s ("
              << 1.0 / model->opt.timestep << " Hz)\n"
              << "Action period: " << actual_control_period << " s ("
              << 1.0 / actual_control_period << " Hz)\n"
              << "Logged transitions: " << transitions.size() << '\n'
              << "Final state: q=" << last.next_q
              << " rad, qvel=" << last.next_qvel << " rad/s\n"
              << "CSV: " << kCsvPath << '\n'
              << "Plot: " << kPlotPath << '\n';
  } catch (const std::exception& exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    mj_deleteData(data);
    mj_deleteModel(model);
    return 1;
  }

  mj_deleteData(data);
  mj_deleteModel(model);
  return 0;
}
