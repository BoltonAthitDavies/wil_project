// Kinematic logistics jobs for gz-sim (Ignition Fortress, gazebo6).
//
// WHY THIS EXISTS
//   KinematicTrajectory moves one prop back and forth along a line. That reads
//   as "boxes drifting", not as a warehouse at work. In a warehouse the moving
//   thing is a CARRIER (a pallet jack, a picker trolley) and the goods move
//   because the carrier takes them: drive to a pallet, slide the forks under,
//   carry it down an aisle to the zone it belongs in, set it down, go for the
//   next one. This system does exactly that, with the same kinematic mechanism
//   as KinematicTrajectory (models stay <static>true</static>, poses are
//   written each step through WorldPoseCmd, so the physics solver never sees
//   them and the camera rate is unaffected). What is given up is the same:
//   no collision response, so a carrier meeting the robot overlaps it.
//
// MODEL
//   A world-level system holding any number of carriers. Each carrier runs a
//   JOB, a list of steps, forever:
//     <goto>x y</goto>                drive in a straight line to (x, y)
//     <goto reverse="true">x y</goto> the same, driving BACKWARDS: the carrier
//                                     keeps facing away from the target.  A jack
//                                     backs out of a slot; it does not spin
//                                     round with a pallet on the forks.
//     <pick load="NAME">yaw</pick>    drive to NAME's current position so that
//                                     the load sits at <carry_offset> in front
//                                     (approach heading yaw), wait
//                                     <handling_time>, attach the load
//     <drop heading="h">x y yaw</drop>
//                                     drive so that the carried load sits at
//                                     (x, y, yaw) with the carrier approaching
//                                     along heading h (defaults to yaw), wait,
//                                     release it there.  The two angles are
//                                     different things: yaw is how the load is
//                                     parked, h is which side the jack comes
//                                     from.  Using one for both put the jack
//                                     on the wrong side of every slot.
//     <wait>seconds</wait>            stand still
//   Motion is turn-in-place at <turn_rate> then straight at <speed>, which is
//   how a pallet jack actually moves. A carried load is re-posed every step
//   from the carrier pose and the relative transform captured at pick-up, so
//   it rides the forks rigidly through turns. <pick> uses the load's CURRENT
//   pose, so a job that keeps moving the same loads between slots stays
//   consistent across loops as long as the slot sequence returns to its start
//   (script/make_realistic_world.py simulates every job to make sure).
//
// SDF
//   <plugin filename="LogisticsScheduler" name="logistics::LogisticsScheduler">
//     <carrier name="model name">
//       <speed>1.3</speed>                   m/s            (default 1.2)
//       <turn_rate>1.5</turn_rate>           rad/s          (default 1.5)
//       <carry_offset>1.6 0 0.0</carry_offset>  load centre in the carrier
//                                            frame when carried (default 1.5 0 0)
//       <handling_time>5</handling_time>     s per pick or drop (default 4)
//       <start>3</start>                     s before the first step (default 0)
//       <job> ...steps... </job>
//     </carrier>
//   </plugin>
//   Carriers and loads are looked up by model name on first update; a name
//   that resolves to nothing disables that carrier with an error.

#include <ignition/plugin/Register.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/gazebo/EntityComponentManager.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/gazebo/components/Model.hh>
#include <ignition/gazebo/components/Pose.hh>
#include <ignition/gazebo/components/PoseCmd.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Quaternion.hh>
#include <ignition/common/Console.hh>
#include <chrono>
#include <map>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace logistics
{
using ignition::gazebo::Entity;
using ignition::gazebo::EntityComponentManager;
using ignition::math::Pose3d;
using ignition::math::Vector3d;
using ignition::math::Quaterniond;

namespace components = ignition::gazebo::components;

struct Step
{
  enum Kind { GOTO, PICK, DROP, WAIT } kind{GOTO};
  double x{0}, y{0}, yaw{0}, heading{0}, seconds{0};
  bool reverse{false};
  std::string load;
};

struct Carrier
{
  std::string name;
  double speed{1.2}, turnRate{1.5}, handling{4.0}, start{0.0};
  Vector3d carryOffset{1.5, 0, 0};
  std::vector<Step> job;

  // runtime
  bool resolved{false}, disabled{false};
  Entity entity{ignition::gazebo::kNullEntity};
  Pose3d pose;                       // our own copy; written every step
  size_t step{0};
  enum Phase { START, TURN, MOVE, ALIGN, HANDLE, DONE } phase{START};
  double timer{0};
  // the target of the current step, resolved when the step begins
  double tx{0}, ty{0}, tyaw{0};
  bool hasFinalYaw{false};
  // carried load
  Entity load{ignition::gazebo::kNullEntity};
  std::string loadName;
  Pose3d loadPose;                   // own copy of the load's pose
  Pose3d rel;                        // load in carrier frame while carried
};

class LogisticsScheduler
    : public ignition::gazebo::System,
      public ignition::gazebo::ISystemConfigure,
      public ignition::gazebo::ISystemPreUpdate
{
  public: void Configure(const Entity &,
                         const std::shared_ptr<const sdf::Element> &_sdf,
                         EntityComponentManager &,
                         ignition::gazebo::EventManager &) override
  {
    auto sdf = _sdf->Clone();
    if (!sdf->HasElement("carrier"))
    {
      ignerr << "LogisticsScheduler: no <carrier> declared; nothing to do." << std::endl;
      return;
    }
    for (auto c = sdf->GetElement("carrier"); c; c = c->GetNextElement("carrier"))
    {
      Carrier k;
      k.name = c->Get<std::string>("name");
      if (c->HasElement("speed")) k.speed = c->Get<double>("speed");
      if (c->HasElement("turn_rate")) k.turnRate = c->Get<double>("turn_rate");
      if (c->HasElement("handling_time")) k.handling = c->Get<double>("handling_time");
      if (c->HasElement("start")) k.start = c->Get<double>("start");
      if (c->HasElement("carry_offset")) k.carryOffset = c->Get<Vector3d>("carry_offset");
      if (c->HasElement("job"))
      {
        auto j = c->GetElement("job");
        for (auto e = j->GetFirstElement(); e; e = e->GetNextElement())
        {
          Step s;
          const std::string tag = e->GetName();
          if (tag == "goto")
          {
            s.kind = Step::GOTO;
            auto v = e->Get<ignition::math::Vector2d>();
            s.x = v.X(); s.y = v.Y();
            if (e->HasAttribute("reverse"))
              s.reverse = e->Get<bool>("reverse");
          }
          else if (tag == "pick")
          {
            s.kind = Step::PICK;
            s.load = e->Get<std::string>("load");
            s.yaw = e->Get<double>();
          }
          else if (tag == "drop")
          {
            s.kind = Step::DROP;
            auto v = e->Get<Vector3d>();
            s.x = v.X(); s.y = v.Y(); s.yaw = v.Z();
            s.heading = e->HasAttribute("heading") ? e->Get<double>("heading") : s.yaw;
          }
          else if (tag == "wait")
          {
            s.kind = Step::WAIT;
            s.seconds = e->Get<double>();
          }
          else
          {
            ignerr << "LogisticsScheduler: carrier " << k.name
                   << ": unknown step <" << tag << ">, ignored." << std::endl;
            continue;
          }
          k.job.push_back(s);
        }
      }
      if (k.job.empty())
      {
        ignerr << "LogisticsScheduler: carrier " << k.name << " has no steps; disabled."
               << std::endl;
        k.disabled = true;
      }
      this->carriers.push_back(k);
    }
  }

  private: static Entity FindModel(EntityComponentManager &_ecm, const std::string &_name)
  {
    return _ecm.EntityByComponents(components::Name(_name), components::Model());
  }

  private: static void WritePose(EntityComponentManager &_ecm, Entity _e, const Pose3d &_p)
  {
    auto cmd = _ecm.Component<components::WorldPoseCmd>(_e);
    if (!cmd)
      _ecm.CreateComponent(_e, components::WorldPoseCmd(_p));
    else
      *cmd = components::WorldPoseCmd(_p);
    _ecm.SetChanged(_e, components::WorldPoseCmd::typeId,
                    ignition::gazebo::ComponentState::OneTimeChange);
  }

  private: static double Wrap(double a)
  {
    while (a > M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    return a;
  }

  // Resolve the current step into a carrier target pose. Returns false when the
  // step cannot be resolved (unknown load), which disables the carrier.
  private: bool BeginStep(Carrier &k, EntityComponentManager &_ecm)
  {
    const Step &s = k.job[k.step];
    k.hasFinalYaw = false;
    switch (s.kind)
    {
      case Step::GOTO:
        k.tx = s.x; k.ty = s.y;
        k.phase = Carrier::TURN;
        return true;
      case Step::WAIT:
        k.timer = s.seconds;
        k.phase = Carrier::HANDLE;
        return true;
      case Step::PICK:
      {
        Entity e = FindModel(_ecm, s.load);
        if (e == ignition::gazebo::kNullEntity)
        {
          ignerr << "LogisticsScheduler: carrier " << k.name << ": load " << s.load
                 << " not found; carrier disabled." << std::endl;
          return false;
        }
        // The load's pose: our copy if we have moved it before, else the ECM's.
        Pose3d lp;
        auto it = this->loadPoses.find(s.load);
        if (it != this->loadPoses.end())
          lp = it->second;
        else
        {
          auto p = _ecm.Component<components::Pose>(e);
          if (!p) return false;
          lp = p->Data();
          this->loadPoses[s.load] = lp;
        }
        k.load = e; k.loadName = s.load; k.loadPose = lp;
        const double c = std::cos(s.yaw), sn = std::sin(s.yaw);
        k.tx = lp.Pos().X() - (c * k.carryOffset.X() - sn * k.carryOffset.Y());
        k.ty = lp.Pos().Y() - (sn * k.carryOffset.X() + c * k.carryOffset.Y());
        k.tyaw = s.yaw; k.hasFinalYaw = true;
        k.phase = Carrier::TURN;
        return true;
      }
      case Step::DROP:
      {
        const double c = std::cos(s.heading), sn = std::sin(s.heading);
        k.tx = s.x - (c * k.carryOffset.X() - sn * k.carryOffset.Y());
        k.ty = s.y - (sn * k.carryOffset.X() + c * k.carryOffset.Y());
        k.tyaw = s.heading; k.hasFinalYaw = true;
        k.phase = Carrier::TURN;
        return true;
      }
    }
    return false;
  }

  // The action at the end of a pick/drop handling wait.
  private: void FinishStep(Carrier &k, EntityComponentManager &_ecm)
  {
    const Step &s = k.job[k.step];
    if (s.kind == Step::PICK && k.load != ignition::gazebo::kNullEntity)
    {
      // capture load-in-carrier so it rides rigidly from here on
      k.rel = k.loadPose - k.pose;        // Pose3d operator-: expressed in k.pose frame
    }
    else if (s.kind == Step::DROP && k.load != ignition::gazebo::kNullEntity)
    {
      Pose3d lp(s.x, s.y, k.loadPose.Pos().Z(), 0, 0, s.yaw);
      k.loadPose = lp;
      this->loadPoses[k.loadName] = lp;
      WritePose(_ecm, k.load, lp);
      k.load = ignition::gazebo::kNullEntity;
      k.loadName.clear();
    }
    k.step = (k.step + 1) % k.job.size();
    k.phase = Carrier::DONE;              // next update begins the next step
  }

  public: void PreUpdate(const ignition::gazebo::UpdateInfo &_info,
                         EntityComponentManager &_ecm) override
  {
    if (_info.paused) return;
    const double dt = std::chrono::duration<double>(_info.dt).count();
    if (dt <= 0) return;

    for (auto &k : this->carriers)
    {
      if (k.disabled) continue;
      if (!k.resolved)
      {
        k.entity = FindModel(_ecm, k.name);
        auto p = k.entity != ignition::gazebo::kNullEntity
                     ? _ecm.Component<components::Pose>(k.entity) : nullptr;
        if (!p)
        {
          ignerr << "LogisticsScheduler: carrier model " << k.name
                 << " not found; disabled." << std::endl;
          k.disabled = true;
          continue;
        }
        k.pose = p->Data();
        k.resolved = true;
        k.timer = k.start;
        k.phase = Carrier::START;
      }

      bool moved = false;
      switch (k.phase)
      {
        case Carrier::START:
          k.timer -= dt;
          if (k.timer <= 0)
          {
            if (!BeginStep(k, _ecm)) k.disabled = true;
          }
          break;

        case Carrier::DONE:
          if (!BeginStep(k, _ecm)) k.disabled = true;
          break;

        case Carrier::TURN:
        {
          const double dx = k.tx - k.pose.Pos().X(), dy = k.ty - k.pose.Pos().Y();
          const double dist = std::hypot(dx, dy);
          if (dist < 1e-3)
          {
            k.phase = k.hasFinalYaw ? Carrier::ALIGN : Carrier::HANDLE;
            if (k.phase == Carrier::HANDLE) k.timer = 0;   // a goto has no wait
            break;
          }
          double want = std::atan2(dy, dx);
          if (k.job[k.step].kind == Step::GOTO && k.job[k.step].reverse)
            want = Wrap(want + M_PI);          // back towards the target
          const double err = Wrap(want - k.pose.Rot().Yaw());
          const double maxTurn = k.turnRate * dt;
          double yaw;
          if (std::fabs(err) <= maxTurn) { yaw = want; k.phase = Carrier::MOVE; }
          else yaw = k.pose.Rot().Yaw() + (err > 0 ? maxTurn : -maxTurn);
          k.pose.Rot() = Quaterniond(0, 0, yaw);
          moved = true;
          break;
        }

        case Carrier::MOVE:
        {
          const double dx = k.tx - k.pose.Pos().X(), dy = k.ty - k.pose.Pos().Y();
          const double dist = std::hypot(dx, dy);
          const double stepLen = k.speed * dt;
          if (dist <= stepLen)
          {
            k.pose.Pos().X(k.tx); k.pose.Pos().Y(k.ty);
            k.phase = k.hasFinalYaw ? Carrier::ALIGN : Carrier::HANDLE;
            if (k.phase == Carrier::HANDLE) k.timer = 0;
          }
          else
          {
            k.pose.Pos().X(k.pose.Pos().X() + dx / dist * stepLen);
            k.pose.Pos().Y(k.pose.Pos().Y() + dy / dist * stepLen);
          }
          moved = true;
          break;
        }

        case Carrier::ALIGN:
        {
          const double err = Wrap(k.tyaw - k.pose.Rot().Yaw());
          const double maxTurn = k.turnRate * dt;
          double yaw;
          if (std::fabs(err) <= maxTurn)
          {
            yaw = k.tyaw;
            k.phase = Carrier::HANDLE;
            k.timer = k.handling;
          }
          else yaw = k.pose.Rot().Yaw() + (err > 0 ? maxTurn : -maxTurn);
          k.pose.Rot() = Quaterniond(0, 0, yaw);
          moved = true;
          break;
        }

        case Carrier::HANDLE:
          k.timer -= dt;
          if (k.timer <= 0) FinishStep(k, _ecm);
          break;
      }

      if (moved)
      {
        WritePose(_ecm, k.entity, k.pose);
        if (k.load != ignition::gazebo::kNullEntity && k.job[k.step].kind != Step::PICK)
        {
          // carried: ride the forks
          k.loadPose = k.rel + k.pose;   // Pose3d operator+: rel expressed in k.pose
          this->loadPoses[k.loadName] = k.loadPose;
          WritePose(_ecm, k.load, k.loadPose);
        }
      }
    }
  }

  private: std::vector<Carrier> carriers;
  // Every load pose this system has set, by model name: the ECM's Pose of a
  // static model is not guaranteed to follow a WorldPoseCmd within the same
  // step, so the scheduler keeps its own record and never reads back.
  private: std::map<std::string, Pose3d> loadPoses;
};
}  // namespace logistics

IGNITION_ADD_PLUGIN(logistics::LogisticsScheduler,
                    ignition::gazebo::System,
                    logistics::LogisticsScheduler::ISystemConfigure,
                    logistics::LogisticsScheduler::ISystemPreUpdate)
IGNITION_ADD_PLUGIN_ALIAS(logistics::LogisticsScheduler, "logistics::LogisticsScheduler")
