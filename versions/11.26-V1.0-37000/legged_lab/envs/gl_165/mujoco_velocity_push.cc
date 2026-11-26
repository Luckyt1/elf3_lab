// GL-165 standing disturbances for the native MuJoCo preview.
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

namespace {
struct Range {
  double low, high;

  bool read(const mjModel* model, int instance, const char* key) {
    const char* value = mj_getPluginConfig(model, instance, key);
    char extra;
    return value && std::sscanf(value, "%lf %lf %c", &low, &high, &extra) == 2 &&
           std::isfinite(low) && std::isfinite(high) && low <= high;
  }

  double sample(std::minstd_rand& rng) const {
    return std::uniform_real_distribution<double>(low, high)(rng);
  }
};

struct Settings {
  Range interval, x, y, yaw;
  int body, joint, torso;
  double control_dt, force_probability, force_n, duration_s;
};

bool read_number(const mjModel* model, int instance, const char* key, double& out) {
  const char* value = mj_getPluginConfig(model, instance, key);
  char extra;
  return value && std::sscanf(value, "%lf %c", &out, &extra) == 1 && std::isfinite(out);
}

int initialize(const mjModel* model, mjData* data, int instance) {
  Settings settings{};
  settings.body = mj_name2id(model, mjOBJ_BODY, "base_link");
  const char* dt = mj_getPluginConfig(model, instance, "control_dt");
  char extra;
  if (settings.body < 1 || model->body_jntnum[settings.body] != 1 || !dt ||
      std::sscanf(dt, "%lf %c", &settings.control_dt, &extra) != 1 ||
      !std::isfinite(settings.control_dt) || settings.control_dt < model->opt.timestep ||
      !settings.interval.read(model, instance, "interval_s") || settings.interval.low <= 0 ||
      !settings.x.read(model, instance, "x") || !settings.y.read(model, instance, "y") ||
      !settings.yaw.read(model, instance, "yaw")) {
    return -1;
  }
  settings.joint = model->body_jntadr[settings.body];
  if (model->jnt_type[settings.joint] != mjJNT_FREE) return -1;
  const char* probability = mj_getPluginConfig(model, instance, "force_probability");
  if (probability && *probability) {
    settings.torso = mj_name2id(model, mjOBJ_BODY, "body_y_link");
    if (settings.torso < 1 || !read_number(model, instance, "force_probability", settings.force_probability) ||
        settings.force_probability < 0 || settings.force_probability > 1 ||
        !read_number(model, instance, "force_n", settings.force_n) || settings.force_n <= 0 ||
        !read_number(model, instance, "duration_s", settings.duration_s) || settings.duration_s <= 0) return -1;
  }
  data->plugin_data[instance] = reinterpret_cast<uintptr_t>(new Settings(settings));
  return 0;
}

double delay(const Settings& settings, std::minstd_rand& rng) {
  return std::ceil(settings.interval.sample(rng) / settings.control_dt) * settings.control_dt;
}

// Keep the due time and RNG seed in mjData so a copied or restored state replays identically.
void reset(const mjModel*, mjtNum* state, void* plugin_data, int) {
  const auto& settings = *static_cast<const Settings*>(plugin_data);
  std::minstd_rand rng(42);
  state[0] = delay(settings, rng);
  state[1] = rng();
  state[2] = state[3] = state[4] = 0;
}

void apply_force(const mjModel* model, mjData* data, int instance, int) {
  const mjtNum* state = data->plugin_state + model->plugin_stateadr[instance];
  if (data->time + 1e-8 >= state[2]) return;
  const auto& settings = *reinterpret_cast<const Settings*>(data->plugin_data[instance]);
  const mjtNum force[3] = {state[3], state[4], 0};
  mj_applyFT(model, data, force, nullptr, data->xipos + 3 * settings.torso, settings.torso, data->qfrc_passive);
}

void advance(const mjModel* model, mjData* data, int instance) {
  mjtNum* state = data->plugin_state + model->plugin_stateadr[instance];
  if (data->time + 1e-8 < state[0]) return;
  const auto& settings = *reinterpret_cast<const Settings*>(data->plugin_data[instance]);
  std::minstd_rand rng(static_cast<unsigned>(state[1]));
  if (settings.force_probability > 0 && std::bernoulli_distribution(settings.force_probability)(rng)) {
    const double angle = std::uniform_real_distribution<double>(0., 2 * std::acos(-1.))(rng);
    state[3] = settings.force_n * std::cos(angle);
    state[4] = settings.force_n * std::sin(angle);
    const double duration = std::max(1., std::round(settings.duration_s / model->opt.timestep)) * model->opt.timestep;
    state[2] = data->time + duration;
    state[0] = data->time + delay(settings, rng);
    state[1] = rng();
    std::printf("[GL-165 torso push] t=%.3f fx=%+.4f fy=%+.4f N duration=%.3f s\n",
                data->time, state[3], state[4], duration);
    std::fflush(stdout);
    return;
  }
  const double dx = settings.x.sample(rng);
  const double dy = settings.y.sample(rng);
  const double yaw = settings.yaw.sample(rng);
  const mjtNum omega_world[3] = {0, 0, yaw};
  mjtNum rotation[9], com_offset_world[3], omega_local[3], com_velocity_from_yaw[3];
  mju_quat2Mat(rotation, data->qpos + model->jnt_qposadr[settings.joint] + 3);
  mju_mulMatVec3(com_offset_world, rotation, model->body_ipos + 3 * settings.body);
  mju_mulMatTVec3(omega_local, rotation, omega_world);
  mju_cross(com_velocity_from_yaw, omega_world, com_offset_world);
  // Isaac sets world COM velocity. A MuJoCo free joint stores world origin
  // linear velocity and body-frame angular velocity.
  mjtNum* velocity = data->qvel + model->jnt_dofadr[settings.joint];
  velocity[0] += dx;
  velocity[1] += dy;
  mju_subFrom3(velocity, com_velocity_from_yaw);
  mju_addTo3(velocity + 3, omega_local);
  state[0] = data->time + delay(settings, rng);
  state[1] = rng();
  std::printf("[GL-165 velocity push] t=%.3f dx=%+.6f dy=%+.6f m/s dyaw=%+.6f rad/s\n",
              data->time, dx, dy, yaw);
  std::fflush(stdout);
}
}  // namespace

mjPLUGIN_LIB_INIT {
  static const char* attributes[] = {"interval_s", "x", "y", "yaw", "control_dt",
                                     "force_probability", "force_n", "duration_s"};
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "gl165.tang_velocity_push";
  plugin.attributes = attributes;
  plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
  plugin.capabilityflags = mjPLUGIN_PASSIVE;
  plugin.nstate = [](const mjModel*, int) { return 5; };
  plugin.init = initialize;
  plugin.destroy = [](mjData* data, int instance) {
    delete reinterpret_cast<Settings*>(data->plugin_data[instance]);
  };
  plugin.reset = reset;
  plugin.compute = apply_force;
  plugin.advance = advance;
  mjp_registerPlugin(&plugin);
}
