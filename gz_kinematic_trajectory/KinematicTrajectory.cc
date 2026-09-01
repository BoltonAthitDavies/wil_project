// Kinematic waypoint follower for gz-sim (Ignition Fortress, gazebo6).
//
// WHY THIS EXISTS
//   TrajectoryFollower drives models with force and torque, so the model must be
//   <static>false</static> -- a real rigid body resting on the floor. Measured in
//   the warehouse world, 30 such bodies cost 3.4x real time: RTF fell from 1.005
//   to 0.294, dragging the 30 Hz cameras down to ~9 Hz. The cost is NOT the
//   motion and NOT the bodies; it is the persistent contact constraints, which
//   the solver re-derives every step. Thirty dynamic props floating in zero
//   gravity (no contacts) ran at RTF 1.006.
//
//   This plugin sidesteps that entirely. The model stays <static>true</static>,
//   so it never enters the constraint solver, and we write its pose directly
//   every step. A static model moved this way still carries its collision with
//   it -- verified by dropping a ball onto a teleported static platform and
//   watching it land on top at the new location.
//
// WHAT YOU GET, AND WHAT YOU GIVE UP
//   Exact constant-speed motion: no bang-bang force control, so no overshoot, no
//   formation shear, no turnaround torque, and identical trajectories on every
//   run -- which matters when comparing VIO pipelines on the same path.
//   What you give up is collision RESPONSE: a pose-driven body has no velocity,
//   so a contact with the robot resolves as overlap. It can shove the robot, or
//   at speed pass through it in one step. Fine for obstacles meant to be avoided.
//
// SDF
//   <waypoints><waypoint>x y</waypoint>...</waypoints>   world metres, >=2
//   <speed>       m/s along the path            (default 1.0)
//   <loop>        true = ping-pong forever      (default true)
//   <dwell>       seconds paused at each end    (default 0)
// Orientation and z are taken from the model's spawn pose and never changed.

#include <ignition/plugin/Register.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/gazebo/components/Pose.hh>
#include <ignition/gazebo/components/PoseCmd.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector2.hh>
#include <memory>
#include <vector>
#include <cmath>

namespace kinematic_trajectory
{
class KinematicTrajectory
    : public ignition::gazebo::System,
      public ignition::gazebo::ISystemConfigure,
      public ignition::gazebo::ISystemPreUpdate
{
  public: void Configure(const ignition::gazebo::Entity &_entity,
                         const std::shared_ptr<const sdf::Element> &_sdf,
                         ignition::gazebo::EntityComponentManager &_ecm,
                         ignition::gazebo::EventManager &) override
  {
    this->model = ignition::gazebo::Model(_entity);
    auto sdf = _sdf->Clone();

    if (sdf->HasElement("waypoints"))
    {
      auto wpsElem = sdf->GetElement("waypoints");
      for (auto e = wpsElem->GetElement("waypoint"); e; e = e->GetNextElement("waypoint"))
        this->wps.push_back(e->Get<ignition::math::Vector2d>());
    }
    if (sdf->HasElement("speed"))
      this->speed = sdf->Get<double>("speed");
    if (sdf->HasElement("loop"))
      this->loop = sdf->Get<bool>("loop");
    if (sdf->HasElement("dwell"))
      this->dwell = sdf->Get<double>("dwell");

    if (this->wps.size() < 2)
    {
      ignerr << "KinematicTrajectory: need at least 2 <waypoint> entries; "
             << "got " << this->wps.size() << ". Plugin disabled." << std::endl;
      this->wps.clear();
      return;
    }
    // Cumulative arc length so we can map elapsed time -> position at constant speed.
    this->cum.push_back(0.0);
    for (size_t i = 1; i < this->wps.size(); ++i)
      this->cum.push_back(this->cum.back() + this->wps[i].Distance(this->wps[i - 1]));
    this->total = this->cum.back();
    if (this->total <= 1e-9 || this->speed <= 0.0)
    {
      ignerr << "KinematicTrajectory: zero path length or speed. Disabled." << std::endl;
      this->wps.clear();
    }
  }

  public: void PreUpdate(const ignition::gazebo::UpdateInfo &_info,
                         ignition::gazebo::EntityComponentManager &_ecm) override
  {
    if (_info.paused || this->wps.empty())
      return;

    if (!this->init)
    {
      // Spawn pose supplies the z and the orientation we hold for the whole run.
      auto p = _ecm.Component<ignition::gazebo::components::Pose>(this->model.Entity());
      if (!p)
        return;
      this->pose0 = p->Data();
      this->init = true;
    }

    const double t = std::chrono::duration<double>(_info.simTime).count();

    // Time to traverse the path once, plus a dwell at each end.
    const double travel = this->total / this->speed;
    double u;                                   // arc length along the path
    if (this->loop)
    {
      const double period = 2.0 * (travel + this->dwell);
      double phase = std::fmod(t, period);
      if (phase < 0) phase += period;
      if (phase < travel)                       // outbound
        u = phase * this->speed;
      else if (phase < travel + this->dwell)    // waiting at the far end
        u = this->total;
      else if (phase < 2.0 * travel + this->dwell)  // return leg
        u = this->total - (phase - travel - this->dwell) * this->speed;
      else                                      // waiting at the start
        u = 0.0;
    }
    else
    {
      u = std::min(t * this->speed, this->total);
    }

    // Locate u on the polyline.
    size_t i = 1;
    while (i + 1 < this->cum.size() && this->cum[i] < u) ++i;
    const double segLen = this->cum[i] - this->cum[i - 1];
    const double f = segLen > 1e-9 ? (u - this->cum[i - 1]) / segLen : 0.0;
    const auto xy = this->wps[i - 1] + (this->wps[i] - this->wps[i - 1]) * f;

    const ignition::math::Pose3d target(xy.X(), xy.Y(), this->pose0.Pos().Z(),
                                        this->pose0.Rot().Roll(),
                                        this->pose0.Rot().Pitch(),
                                        this->pose0.Rot().Yaw());

    // Same mechanism the set_pose service uses: write WorldPoseCmd and mark it
    // changed. Works on <static> models, and the collision moves with it.
    auto cmd = _ecm.Component<ignition::gazebo::components::WorldPoseCmd>(
        this->model.Entity());
    if (!cmd)
    {
      _ecm.CreateComponent(this->model.Entity(),
          ignition::gazebo::components::WorldPoseCmd(target));
    }
    else
    {
      *cmd = ignition::gazebo::components::WorldPoseCmd(target);
    }
    _ecm.SetChanged(this->model.Entity(),
        ignition::gazebo::components::WorldPoseCmd::typeId,
        ignition::gazebo::ComponentState::OneTimeChange);
  }

  private: ignition::gazebo::Model model{ignition::gazebo::kNullEntity};
  private: std::vector<ignition::math::Vector2d> wps;
  private: std::vector<double> cum;
  private: double total{0.0};
  private: double speed{1.0};
  private: double dwell{0.0};
  private: bool loop{true};
  private: bool init{false};
  private: ignition::math::Pose3d pose0;
};
}  // namespace kinematic_trajectory

IGNITION_ADD_PLUGIN(kinematic_trajectory::KinematicTrajectory,
                    ignition::gazebo::System,
                    kinematic_trajectory::KinematicTrajectory::ISystemConfigure,
                    kinematic_trajectory::KinematicTrajectory::ISystemPreUpdate)
IGNITION_ADD_PLUGIN_ALIAS(kinematic_trajectory::KinematicTrajectory,
                          "kinematic_trajectory::KinematicTrajectory")
