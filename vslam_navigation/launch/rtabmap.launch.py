"""RTAB-Map as a loop-closure back-end behind VINS or ORB-SLAM3, on the sim bags.

Starts two things and nothing else:

  rtab_camera_shim.py   decompresses the bag's stereo pair and manufactures the
                        camera_info + static TF that the bags do not carry
  rtabmap_slam/rtabmap  the mapper, taking odometry from whichever front-end you
                        are running in another terminal

There is deliberately NO odometry node here. RTAB-Map's own visual odometry
(rtabmap_odom) is the thing we are replacing -- the rtabmap node always takes
its pose from the `odom` topic, so simply not launching an odometry node is
what puts VINS or ORB-SLAM3 in that role. (`visual_odometry:=false` is an
argument of the upstream rtabmap_launch wrapper, not a parameter of this node;
passing it here would do nothing.)

Like every other launch file in this workspace, this one does NOT play a bag.

    # terminal 1   front-end
    ros2 launch vins_fusion_ros2 vins_fusion_ros2.launch.py
    # terminal 2   this
    ros2 launch ./rtab_sim.launch.py
    # terminal 3   data
    ros2 bag play dataset/static_dataset_loop --rate 0.5

NOTE the missing --clock. These bags RECORD /clock (74180 messages), so adding
--clock makes rosbag2 publish a second, wall-epoch clock on top of it. The two
disagree by ~1.79e9 s, tf2 logs "Detected jump back in time" and clears its
buffer thousands of times, and RTAB-Map never resolves a camera transform.

For ORB-SLAM3 instead, override the two frame/topic arguments:

    ros2 launch vslam_navigation rtabmap.launch.py \
        odom_frame:=orbslam3_world body_frame:=base_footprint \
        output_path:=output/output_orb_rtab/simulation/dataset_dynamic_000
"""
import glob
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WORKSPACE = os.path.expanduser('~/wil_project')
# Moved into the vslam_navigation package on 2026-11-04 (it lived at the root of
# the rtabmap_ros fork, which is upstream code pinned to one commit). The shim
# and the writer are installed as programs, the RViz config as a share file;
# with --symlink-install they are links to the source tree.
_PKG = 'vslam_navigation'
_LIB = os.path.join(get_package_prefix(_PKG), 'lib', _PKG)
SHIM = os.path.join(_LIB, 'rtab_camera_shim.py')
WRITER = os.path.join(_LIB, 'rtab_trajectory_writer.py')
RVIZ_CFG = os.path.join(get_package_share_directory(_PKG), 'rviz', 'rtab_map.rviz')
CONFIG_DIR = os.path.join(WORKSPACE, 'vins_fusion_ros2', 'config', 'wil_sim')


def _mode_params(mode, read_only, start_pose='origin'):
    """The ~6 parameters that separate the three modes.

    Everything else in the node is shared, and the shared block is where the
    measured comment blocks live (the Grid/RangeMax sweep, the topic_queue_size
    derivation, the extrinsics-bug history). Forking this file per mode would mean
    three copies of those to drift apart, so the modes are a dict merge instead.

    construct  build a map from nothing. Reproduces the pre-mode defaults exactly.
    extend     open an existing map and add a new session to it. RTAB-Map appends
               by default -- `-d` is purely a file delete -- and the new session is
               stitched to the old one by ordinary appearance-based loop closure.
    localize   read-only against an existing map. No nodes are added; the output is
               map->odom and /localization_pose.
    """
    if mode == 'construct':
        return {'Mem/IncrementalMemory': 'true',
                'Mem/InitWMWithAllNodes': 'false'}

    if mode == 'extend':
        return {
            'Mem/IncrementalMemory': 'true',
            # Without this, working memory is initialised with the PREVIOUS
            # session only (Parameters.h:225), so a third session could attach to
            # the second but never to the first. Extending a map means being able
            # to recognise anywhere in it.
            'Mem/InitWMWithAllNodes': 'true',
        }

    # localize
    params = {
        # The master switch: false means Localization mode (Parameters.h:215).
        'Mem/IncrementalMemory': 'false',
        'Mem/InitWMWithAllNodes': 'true',
        # Publish the grid stored in the database rather than re-assembling it.
        'use_saved_map': True,
        # WHERE THE RUN IS ASSUMED TO START. Default 'origin' (user constraint,
        # 11 Nov 2026: the robot is returned to its spawn point before every
        # relocalisation run), which is the right seed here for two measured
        # reasons.
        #
        # First, the seed IS the spawn point on these priors. RTAB-Map logs what
        # it does here -- "last localization pose is ignored
        # (RGBD/StartAtOrigin=true), assuming we start at the first node of the
        # map" -- so the seed is node 1, not the map frame's identity. Both
        # mapping recordings begin at the same spawn pose and node 1 is added
        # 0.8-1.2 s in, by which time the robot has moved 0.000 m on the U layout
        # and 0.051 m on the I layout. The seed is therefore the spawn pose to
        # within 5 cm. (The map frame's identity is a further 0.26 m away, being
        # where the front end initialised; it is not what this parameter uses.)
        #
        # Second, 'false' is actively bad here, which was not obvious. We set
        # Mem/LocalizationReadOnly below, and RTAB-Map documents that the last
        # localisation pose is then never written back, so the run restarts from
        # the pose stored in the database -- wherever the 300 s MAPPING run
        # finished, metres away and in the wrong aisle. The appearance-based
        # relocaliser has to recover from there before any pose is usable.
        #
        # Set start_pose:=last to restore the old behaviour, or give six numbers
        # "x y z roll pitch yaw" to seed an explicit pose in the map frame.
        'RGBD/StartAtOrigin': 'true' if start_pose == 'origin' else 'false',
        # Turns on the LocalizationStatus diagnostic, which only runs when
        # Mem/IncrementalMemory is false. Gives a one-line pass/fail on
        # /diagnostics instead of eyeballing covariances.
        'loc_thr': 0.2,

        # Do not build local occupancy grids while localizing. The upstream
        # default is FALSE; the ROS wrapper forces it true (CoreNode.cpp:45,
        # CoreWrapper.cpp:339) because a mapping node needs it -- but in
        # localization mode /map is served from the database (use_saved_map
        # above), so every grid built here is thrown away.
        #
        # It is not free work. Building a grid runs dense stereo (SGBM) on the
        # pair, measured at 464 ms per keyframe, and that load is what starves
        # VINS: with rtabmap + shim + republish alongside, /imu arrived at 51 Hz
        # against a nominal 60 and VINS diverged to 1595 m in a 14 x 21 m
        # building. Slowing the bag to --rate 0.3 made it WORSE, so the problem
        # is total CPU, not pacing. Feature extraction for loop closure is
        # unaffected -- relocalization still works.
        'RGBD/CreateOccupancyGrid': 'false',
    }
    if read_only:
        # Protects the prebuilt map, but RTAB-Map documents it as incompatible with
        # disjoint sessions in working memory -- a database from a failed `extend`
        # whose sessions never merged will fail to open this way.
        params['Mem/LocalizationReadOnly'] = 'true'
    return params


def _loop_closure_params(enabled):
    """loop_closure:=false turns RTAB-Map into a pure odometry chain.

    Two detectors can add a non-neighbour link: the appearance-based one (Bayes
    filter over visual words, accepted when the hypothesis exceeds
    Rtabmap/LoopThr) and the proximity one (RGBD/ProximityBySpace, nodes close
    in space in the working memory). Both are shut here, so the graph is exactly
    the front-end's odometry and the grid is assembled from it uncorrected.

    Visual words are still EXTRACTED (Kp/MaxFeatures untouched) so the database
    keeps its signatures and remains usable for mode:=localize and for
    rtabmap-reprocess. The cheaper alternative, 'Kp/MaxFeatures': '-1', disables
    extraction altogether and yields a database that can never relocalise.

    Why want this: on the realistic mapping bags the front-end's own closures
    were what corrupted an otherwise ~0.1 m map (see loopClosing: 0 in the ORB
    config and MEMORY_CLAUDE). With both front-end and back-end closures off the
    map is a direct picture of odometry drift, the A/B against the stitched map.
    """
    if enabled:
        # Explicit values, NOT an empty dict. RTAB-Map restores its parameter set
        # FROM THE DATABASE, so a prior built with loop_closure:=false comes back
        # with LoopThr 1.0 and ProximityBySpace false even when this launch is
        # asked for closures. In mode:=localize that is fatal and silent, because
        # relocalising against a prior IS accepting a loop closure against it:
        # the node comes up, publishes /localization_pose, and never localises.
        #
        # Measured on the real rig, 6 Nov 2026, which is what exposed this:
        #   map_prior_v2    (0 closures in the db)  all 76 poses cov_trace 3e+04,
        #                                           RTAB-Map's no-information value
        #   realv7_001_prior (32 closures in the db) cov_trace median 0.022
        # The launch already refuses mode:=localize with loop_closure:=false, so
        # the intent was there; it just did not defend against the database
        # putting the closure-free values back. rosparam wins over the stored
        # value, hence setting them here.
        #
        # 0.11 and true are RTAB-Map's own defaults for these two.
        return {
            'Rtabmap/LoopThr': '0.11',
            'RGBD/ProximityBySpace': 'true',
        }
    return {
        # Bayes posterior is a probability; a threshold of 1.0 is never crossed.
        'Rtabmap/LoopThr': '1.0',
        # 'Kp/MaxFeatures': '-1',   # option A: no words at all, cannot localize
        'RGBD/ProximityBySpace': 'false',
    }


def _geometric_check_params(enabled, correspondence_ratio, max_rotation):
    """Verify an accepted loop closure against the 3-D structure, not just the 2-D
    features that proposed it.

    Reg/Strategy is 0 (Vis) by RTAB-Map default and this launch never set it, so
    every closure in the twelve sweep runs of 11-12 Nov 2026 was accepted on
    feature correspondences alone -- twenty PnP inliers at 2 px reprojection
    error -- and the depth stored against all 148 prior nodes was never consulted.

    WHY THIS IS A DIFFERENT TEST from everything else in the sweep. RGBD/
    OptimizeMaxError and RGBD/MaxOdomCacheSize both judge the GRAPH RESIDUAL, and
    the odometry links in that graph are relative: rotating and translating the
    whole cached chain rigidly leaves their residual near zero, so a wrong
    placement of the entire window is invisible to them. That is why cache size
    was inert (and harmful at 20/30) and why the threshold only halved the warp
    rate. Structural agreement is an ABSOLUTE constraint and has no such null
    space -- it asks whether the racks in view line up with the racks the map
    puts there.

    WHY IT SHOULD WORK HERE. The rotational error of four false fixes, each
    measured against the correct correction that replaced it:
        sweep_B2   73.6 deg -> 0.3 deg      73 deg error
        sweep_B3 -138.6 deg -> 4.0 deg     143 deg
        sweep_A3   22.8 deg -> -0.4 deg     23 deg
        sweep_A2  -24.2 deg -> -0.5 deg     24 deg
    An aisle grid is symmetric at 0 and 180 deg and under aisle-pitch
    translations. None of those four is a symmetry, so the structure cannot
    agree: ICP either fails the correspondence ratio or needs a correction past
    Icp/MaxRotation, and the closure is rejected either way.

    WHAT IT CANNOT DO. A true 180 deg flip, or a translation by exactly one aisle
    pitch, IS a symmetry, and over one local cloud the geometry genuinely agrees.
    Expect the 23/24/73 deg class to go and a residue near 180 deg to survive
    (B2's 143 and H1's 177 are in that grey zone). Bounding WHERE a match may
    come from (proximity-only) or HOW FAST the pose may change (an innovation
    gate) is what finishes those; the three compose rather than compete.

    COST. There are no laser scans in the prior -- 148 nodes with image, depth
    and calibration -- so this runs ICP on the depth cloud per candidate closure,
    on a laptop that throttles at 90 C. Expect CPU up and the acceptance rate
    down.
    """
    if not enabled:
        return {}
    return {
        'Reg/Strategy': '2',                 # Vis to propose, ICP to verify
        'Icp/CorrespondenceRatio': correspondence_ratio,
        'Icp/MaxRotation': max_rotation,
        # RAISED from RTAB-Map's 0.2 m, deliberately. 0.2 is tight enough to
        # reject an honest closure that needs a modest refinement, which would
        # starve acceptance the way odom_cache_size 20/30 did.
        'Icp/MaxTranslation': '0.5',
        # Stereo depth is noisy at warehouse range, so plane normals estimated
        # from it are unreliable and 0.05 m correspondences find nothing.
        'Icp/PointToPlane': 'false',
        'Icp/MaxCorrespondenceDistance': '0.3',
        'Icp/VoxelSize': '0.1',
    }


def launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration('mode').perform(context).lower()
    if mode not in ('construct', 'extend', 'localize'):
        raise RuntimeError(
            f"mode must be construct|extend|localize, got {mode!r}")
    camera_model = LaunchConfiguration('camera_model').perform(context).lower()
    if camera_model not in ('pinhole', 'fisheye'):
        raise RuntimeError(f"camera_model must be pinhole|fisheye, got {camera_model!r}")

    # start_pose: 'origin' (the spawn point, the default), 'last' (the pose left
    # in the database), or six numbers "x y z roll pitch yaw" in the MAP frame.
    # RTAB-Map applies both mechanisms in localization mode only, so asking for a
    # seed while building or extending a map is refused rather than ignored.
    start_pose = LaunchConfiguration('start_pose').perform(context).strip().lower()
    initial_pose = ''
    if start_pose not in ('origin', 'last'):
        parts = start_pose.replace(',', ' ').split()
        if len(parts) != 6:
            raise RuntimeError(
                f"start_pose must be 'origin', 'last', or six numbers "
                f'"x y z roll pitch yaw"; got {start_pose!r}')
        try:
            [float(v) for v in parts]
        except ValueError:
            raise RuntimeError(f'start_pose is not six numbers: {start_pose!r}')
        initial_pose = ' '.join(parts)
    if mode != 'localize' and start_pose != 'origin':
        raise RuntimeError(
            f"start_pose:={start_pose!r} applies to mode:=localize only "
            f'(RTAB-Map ignores it while building or extending a map), got mode {mode!r}')
    odom_frame = LaunchConfiguration('odom_frame').perform(context)
    body_frame = LaunchConfiguration('body_frame').perform(context)
    # frontend:=orb|vins picks the frame pair the front end broadcasts, so the
    # four frame arguments need not be typed. ORB-SLAM3 publishes to
    # base_footprint in simulation (wil_sim_stereo_imu.launch.py) but to
    # imu_link on the real rig (wil_stereo_imu.launch.py); VINS uses world->body
    # in both domains. Explicit odom_frame/body_frame win when frontend is empty.
    frontend = LaunchConfiguration('frontend').perform(context).lower()
    if frontend == 'orb':
        odom_frame = 'orbslam3_world'
        body_frame = 'imu_link' if camera_model == 'fisheye' else 'base_footprint'
    elif frontend == 'vins':
        odom_frame, body_frame = 'world', 'body'
    elif frontend:
        raise RuntimeError(f"frontend must be orb|vins (or empty), got {frontend!r}")
    # On the real rig the front ends' body frame is the IMU frame, which is
    # x-right, y-up, z-backward (ORB IMU.T_b_c1 = VINS body_T_cam0 = diag(1,-1,-1)),
    # and RTAB-Map measures ground height along base_frame z. So a forward-left-up
    # base_link is published under the body frame here, instead of in a separate
    # terminal: rotation Rz(90)Ry(90) (base forward = -z_imu, left = -x_imu,
    # up = +y_imu, verified numerically 2026-10-29), translation (0, -h, 0) with
    # h the IMU height above the floor (measured 0.11 m by the user, 2026-11-04).
    # If the IMU extrinsic in either front-end config changes, this rotation
    # must change with it. In simulation the body frames are already FLU, so
    # nothing is published (publish_base_tf:=auto).
    publish_base_tf = LaunchConfiguration('publish_base_tf').perform(context).lower()
    if publish_base_tf in ('', 'auto'):
        publish_base_tf = camera_model == 'fisheye'
    else:
        publish_base_tf = publish_base_tf in ('true', '1', 'yes')
    imu_height = float(LaunchConfiguration('imu_height').perform(context))
    base_frame = LaunchConfiguration('base_frame').perform(context) or (
        'base_link' if publish_base_tf else body_frame)
    ground_height = LaunchConfiguration('ground_height').perform(context)
    optimize_max_error = LaunchConfiguration('optimize_max_error').perform(context)
    min_inliers = LaunchConfiguration('min_inliers').perform(context)
    min_inliers_distribution = LaunchConfiguration('min_inliers_distribution').perform(context)
    odom_cache_size = LaunchConfiguration('odom_cache_size').perform(context)
    geometric_check = LaunchConfiguration('geometric_check').perform(context).lower() \
        in ('true', '1', 'yes')
    icp_correspondence_ratio = LaunchConfiguration('icp_correspondence_ratio').perform(context)
    icp_max_rotation = LaunchConfiguration('icp_max_rotation').perform(context)
    estimation_type = LaunchConfiguration('estimation_type').perform(context)
    vis_inlier_distance = LaunchConfiguration('vis_inlier_distance').perform(context)
    gen_scan = LaunchConfiguration('gen_scan').perform(context).lower() \
        in ('true', '1', 'yes')
    localize_delay_s = float(LaunchConfiguration('localize_delay_s').perform(context))
    loop_closure = LaunchConfiguration('loop_closure').perform(context).lower() in (
        'true', '1', 'yes')

    # relocalize:=false is DEAD RECKONING ON A PRIOR MAP: hold the pose at the
    # seed composed with the front end's odometry, serve the prior's grid, and
    # never correct. Asked for 11 Nov 2026 ("just using the initial pose and
    # vslam pose"). It is the same mechanism as loop_closure:=false -- the
    # appearance detector's threshold is put out of reach and proximity
    # detection is off -- so it is expressed as its own argument to keep the
    # intent legible, and it relaxes the refusal below.
    relocalize = LaunchConfiguration('relocalize').perform(context).lower() in (
        'true', '1', 'yes')
    if not relocalize:
        if mode != 'localize':
            raise RuntimeError(
                f'relocalize:=false applies to mode:=localize only, got mode {mode!r}')
        if start_pose == 'last':
            # Dead reckoning has no way back from a wrong start, and the stored
            # pose is where the MAPPING run ended, not where this one begins.
            raise RuntimeError(
                "relocalize:=false needs a known start: use start_pose:=origin (the "
                "point the map was collected from) or an explicit \"x y z r p y\". "
                "start_pose:=last would silently begin at the mapping run's end.")
        loop_closure = False
    if mode == 'localize' and not loop_closure and relocalize:
        # Relocalisation IS the appearance-based detector: Rtabmap/LoopThr 1.0
        # would make it impossible to ever localise against the prior map. Pass
        # relocalize:=false if that is what you actually want.
        raise RuntimeError('mode:=localize needs loop_closure:=true '
                           '(relocalisation uses the loop-closure detector), '
                           'or relocalize:=false to dead reckon from the seed')
    read_only = LaunchConfiguration('read_only').perform(context).lower() in (
        'true', '1', 'yes')
    database = os.path.expanduser(LaunchConfiguration('database_path').perform(context))
    output_path = LaunchConfiguration('output_path').perform(context)
    # delete_db defaults to '' meaning "whatever the mode implies": only
    # `construct` starts from nothing. An explicit true on extend/localize is
    # refused rather than honoured -- both exist to work against a map that was
    # expensive to build, and -d would silently erase it before the first frame.
    delete_db_arg = LaunchConfiguration('delete_db').perform(context).lower()
    if delete_db_arg in ('', 'auto'):
        delete_db = (mode == 'construct')
    else:
        delete_db = delete_db_arg in ('true', '1', 'yes')
        if delete_db and mode != 'construct':
            raise RuntimeError(
                f"delete_db:=true would erase the database that mode:={mode} "
                f"exists to read. Use mode:=construct to start fresh.")
    use_rviz = LaunchConfiguration('rviz').perform(context).lower() in (
        'true', '1', 'yes')
    always_update = LaunchConfiguration('map_always_update').perform(context).lower() in (
        'true', '1', 'yes')
    wait_tf = float(LaunchConfiguration('wait_for_transform').perform(context))
    grid_range_max = LaunchConfiguration('grid_range_max').perform(context)
    detection_rate = LaunchConfiguration('detection_rate').perform(context)
    shim_rate = LaunchConfiguration('shim_rate').perform(context)
    config_dir = LaunchConfiguration('config_dir').perform(context)
    left_cam = LaunchConfiguration('left_cam').perform(context)
    hfov_deg = LaunchConfiguration('hfov_deg').perform(context)
    rect_scale = LaunchConfiguration('rect_scale').perform(context)

    # database_path must name a FILE. RTAB-Map is launched with -d, which erases
    # whatever that path names before starting, so pointing it at a run directory
    # (easy to do when it looks just like output_path) puts a delete on a
    # directory holding results. Resolve anything that is not a .db to a file
    # INSIDE it, so -d can only ever remove the database itself.
    if os.path.isdir(database) or not database.lower().endswith('.db'):
        # A directory holding exactly one .db is almost always that file
        # (2026-11-04: `database_path:=map/realv7_001_lc` in extend mode opened a
        # NEW empty rtabmap.db beside realv7_001_lc.db, and the user saw no map).
        inside = sorted(glob.glob(os.path.join(database, '*.db')))
        if len(inside) == 1:
            database = inside[0]
            print(f'[rtab_sim] database_path is a directory with one database; using {database}')
        else:
            database = os.path.join(database, 'rtabmap.db')
            print(f'[rtab_sim] database_path was not a .db file; using {database}')
    if mode in ('extend', 'localize') and not os.path.isfile(database):
        raise RuntimeError(
            f"mode:={mode} needs an existing database to read, but {database} does not "
            f"exist. Give the .db file itself (database_path:=.../name.db) or use "
            f"mode:=construct to start a new map.")
    # A prior with no VOCABULARY cannot be relocalised against, and the symptom
    # is silent: the node starts, publishes /localization_pose, and every pose
    # carries RTAB-Map's no-information covariance (trace 3e4). Checked here
    # because it is cheap and the alternative is discovering it from a run.
    #
    # The Feature table stores each keypoint's word_id, but the Word table stores
    # the word DESCRIPTORS, and only the latter can be matched against. Measured
    # 11 Nov 2026: dataset_realv7_map's features reference 35,683 distinct word
    # ids while its Word table holds 122 rows, so the run that used it localised
    # nothing. Both priors built that night are like this; everything built
    # 10 Nov 06:45 or earlier is intact, including closure-free ones, so this is
    # not a consequence of loop_closure:=false.
    if mode == 'localize' and relocalize:   # nothing is matched when relocalize is off
        try:
            import sqlite3
            _c = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
            _w = _c.execute('select count(*) from Word').fetchone()[0]
            _d = _c.execute('select count(distinct word_id) from Feature '
                            'where word_id>0').fetchone()[0]
            _c.close()
        except Exception:
            _w = _d = None          # unreadable: let RTAB-Map report it instead
        if _w is not None and _d and _w < 0.2 * _d:
            raise RuntimeError(
                f'{database} has no usable visual vocabulary: {_w} Word rows against '
                f'{_d} distinct word ids referenced by its features.\n'
                f'  Relocalisation matches against the Word table, so this prior cannot '
                f'be localised against;\n'
                f'  every pose would carry covariance trace 3e4 and no fix would ever be '
                f'accepted.\n'
                f'  Rebuild the prior, or use one whose vocabulary is intact.')

    # geometric_check verifies a closure by ICP, and RTAB-Map's ICP registers the
    # LASER SCAN of each node -- not its depth image. Nothing in this stack ever
    # produced a scan: gen_scan is a node parameter defaulting to false, so all
    # 148 nodes of dataset_allsensor_realistic_map carry image, depth and
    # calibration but zero scan rows. Reg/Strategy 2 against that prior leaves
    # ICP with nothing to register, so EVERY closure fails verification and the
    # run localises nothing at all -- measured 12 Nov 2026, two runs, both with
    # no informative pose in the whole run, the same signature as
    # min_inliers_distribution 0.03. I checked that the depth was present and
    # concluded ICP was feasible; the depth is not what ICP reads.
    if mode == 'localize' and relocalize and geometric_check:
        try:
            import sqlite3
            _c = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
            _s = _c.execute('select count(*) from Data where scan is not null '
                            'and length(scan)>0').fetchone()[0]
            _n = _c.execute('select count(*) from Node').fetchone()[0]
            _c.close()
        except Exception:
            _s = _n = None
        if _s is not None and _n and _s < 0.5 * _n:
            raise RuntimeError(
                f'geometric_check:=true needs laser scans in the prior, and '
                f'{os.path.basename(database)} has {_s} of {_n} nodes with one.\n'
                f'  RTAB-Map ICP registers Data.scan, not Data.depth, so every closure '
                f'would fail verification\n'
                f'  and the run would localise nothing. Generate the scans from the depth '
                f'already stored:\n\n'
                f'    rtabmap-reprocess -scan_from_depth \\\n'
                f'        "{database}" "{database[:-3]}_scan.db"\n\n'
                f'  then localise against that copy, and pass gen_scan:=true so the live '
                f'frames carry a scan too.\n'
                f'  Both sides need one. Alternatively use estimation_type:=0, which '
                f'checks the SPARSE\n'
                f'  feature geometry and needs no scan at all.')
    os.makedirs(os.path.dirname(database), exist_ok=True)
    print(f'[rtab_sim] mode     = {mode}'
          + ('  (read-only)' if (read_only and mode == 'localize') else ''))
    print(f'[rtab_sim] odom     = TF {odom_frame} -> {body_frame}'
          + (f'  (frontend:={frontend})' if frontend else ''))
    if publish_base_tf:
        print(f'[rtab_sim] base TF  = {body_frame} -> {base_frame} published here, '
              f'IMU height {imu_height:g} m')
    if not loop_closure:
        print('[rtab_sim] loop closure OFF: pure odometry chain (LoopThr 1.0, '
              'ProximityBySpace false; visual words still extracted)')
    if mode == 'localize' and not relocalize:
        print('[rtab_sim] RELOCALISATION OFF: the pose is the start_pose seed plus '
              'front-end odometry.')
        print('[rtab_sim]   No correction will ever be published, so drift is '
              'unbounded and grows with')
        print('[rtab_sim]   distance travelled. corr_step_m should read 0.000 for '
              'every sample; if it')
        print('[rtab_sim]   does not, something else is publishing map -> odom.')
    print(f'[rtab_sim] base     = {base_frame}   ground_height = {ground_height}')
    print(f'[rtab_sim] body     = {body_frame}')
    print(f'[rtab_sim] cameras  = {camera_model}  config_dir = {config_dir}'
          + (f'  left_cam = {left_cam}  hfov = {hfov_deg} deg  rect_scale = {rect_scale}'
             if camera_model == 'fisheye' else ''))
    print(f'[rtab_sim] database = {database}')
    print(f'[rtab_sim] output   = {output_path or "(none -- no vio.csv written)"}')

    shim = ExecuteProcess(
        cmd=['python3', SHIM,
             '--ros-args',
             '-p', 'use_sim_time:=true',
             '-p', f'body_frame:={body_frame}',
             '-p', f'config_dir:={config_dir}',
             '-p', f'rate:={shim_rate}',
             '-p', f'camera_model:={camera_model}',
             '-p', f'left_cam:={left_cam}',
             # Always hand these over as floats: a bare '110' would be parsed
             # as an integer and rejected by the shim's DOUBLE parameter.
             '-p', f'hfov_deg:={float(hfov_deg)}',
             '-p', f'rect_scale:={float(rect_scale)}',
             '-p', 'namespace:=/stereo'],
        output='screen')

    rtabmap = Node(
        package='rtabmap_slam', executable='rtabmap', name='rtabmap',
        output='screen',
        # -d wipes the database at startup. Without it RTAB-Map appends to the
        # previous run's graph and the second run's loop closures are meaningless.
        # Set delete_db:=false to accumulate instead.
        arguments=['-d'] if delete_db else [],
        parameters=[{
            'use_sim_time': True,
            # The robot frame RTAB-Map reports poses in. base_footprint (not the
            # IMU body) is what nav2 and localization_pose must agree on. Grid
            # heights are measured in THIS frame -- see ground_height below.
            'frame_id': base_frame,
            # Take the pose from TF rather than from an odometry TOPIC. Both
            # front-ends broadcast it (vins_estimator.cpp sendTransform, and
            # orbslam3_node's publish_tf, on by default), and it is the only
            # arrangement that syncs reliably here: odometry runs at 40 Hz+
            # while the stereo pair is throttled to ~3 Hz, and folding that
            # into RTAB-Map's 5-topic ApproximateTime set starves it -- it
            # reported "Did not receive data" and kept 3 nodes out of a 45 s
            # run. Using TF leaves only the four image/camera_info topics in
            # the sync, and those share a timestamp exactly. Same run, TF:
            # 28 nodes, 25 neighbour links, 2 loop closures, no warnings.
            # The cost is losing the odometry covariance RTAB-Map would
            # otherwise weight its graph edges with.
            'odom_frame_id': odom_frame,
            'database_path': database,

            'subscribe_stereo': True,
            'subscribe_depth': False,
            'subscribe_rgb': False,

            # The bag's cam0/cam1 stamps differ by ~15 ms and odometry arrives on
            # its own schedule, so nothing here is exactly synchronised.
            'approx_sync': True,
            'sync_queue_size': 30,
            # The per-topic input queue, NOT the sync queue. It must be deep
            # enough to still hold an odometry message when its stereo pair
            # finally shows up: odometry runs at 40 Hz+ while the images are
            # throttled to ~3 Hz, so the default 10 evicts the match in 0.25 s
            # and RTAB-Map reports "Did not receive data since 5 seconds".
            'topic_queue_size': 200,

            # How long to hold an image waiting for the odometry TF that covers
            # its stamp. The camera extrinsics are static, so this is entirely
            # slack for the FRONT-END's latency -- and a front-end that cannot
            # keep up with playback runs a persistent lag, not a jittery one.
            # Measured with VINS at proc_ms ~114 against 100 ms frame spacing
            # (bag --rate 0.5): a steady 83-frame backlog, i.e. a ~4 s offset,
            # against which the old 0.3 s dropped essentially every frame. The
            # images are throttled to 4 Hz with topic_queue_size 200, so there is
            # ~50 s of buffer to wait inside; being generous here costs nothing.
            'wait_for_transform': wait_tf,

            # One loop-closure check per second. The default 1 Hz is already
            # right for a 93 s run at walking pace; raise it for faster motion.
            # Keeps the assembled-map cache even with no subscribers
            # (CoreWrapper.cpp:4180 drops it otherwise). Measured: NOT needed for
            # a late-attaching RViz -- a subscriber joining 32 s into a run still
            # got the full map (492x811 grid, 113k points), because
            # updateMapCaches() rebuilds from Memory. Left available for cases
            # where that rebuild is too slow, not as a fix for late attach.
            'map_always_update': always_update,

            # --- stereo depth quality -------------------------------------
            # Dense stereo defaults to cv::StereoBM, which needs local texture
            # and has none to work with on a painted warehouse floor or a blank
            # wall -- it emits mismatches strung out along the viewing ray, which
            # is what the vertical smears in the cloud are. SGBM is semi-global
            # and far more robust on exactly this kind of surface.
            'Stereo/DenseStrategy': '1',            # 0=StereoBM, 1=StereoSGBM

            # Depth precision is fx*B / disparity, and this rig is fx=640 with a
            # 93.6 mm baseline, so fx*B = 59.9 px*m, so a ONE-pixel disparity
            # error is 0.42 m at 5 m and 1.67 m at 10 m. That argument was used
            # to justify a 4.0 m cap -- but MEASURED against the baked ground
            # truth, the far points are mostly real and cropping them costs more
            # coverage than it buys accuracy:
            #
            #   range   precision   recall     F1   chamfer p95
            #    4 m      94.0 %    44.8 %   60.6      0.30 m
            #    6 m      90.0 %    41.5 %   56.8      0.35 m
            #    8 m      86.0 %    49.8 %   63.1      0.42 m   (mean of 2 runs)
            #   10 m      89.4 %    50.9 %   64.9      0.35 m   <- default
            #   12 m      87.9 %    52.2 %   65.5      0.40 m
            #
            # Caveat: rtabmap-reprocess is NOT deterministic -- it re-runs loop
            # detection and gave 129 vs 133 closures on identical input, worth
            # about 1 F1 point. Differences smaller than that are noise.
            'Grid/RangeMax': grid_range_max,

            # --- ground / obstacle split ----------------------------------
            # By default RTAB-Map decides ground-vs-obstacle from point NORMALS
            # (Grid/NormalsSegmentation) against a 45 deg cone. Normals estimated
            # on noisy stereo depth over a flat, low-texture warehouse floor are
            # meaningless, so the floor itself gets labelled obstacle: measured
            # /cloud_ground 146 points against /cloud_obstacles 76400, which is
            # the scatter of black obstacle cells all over the map.
            #
            # The robot drives a known-flat floor, so a plain height threshold is
            # both simpler and far more robust. The docs say MaxGroundHeight
            # "should be set if NormalsSegmentation is false" -- ground truth puts
            # the body frame ~0.02 m above the floor, so 0.1 m is generous.
            'Grid/NormalsSegmentation': 'false',
            # Grid/MapFrameProjection defaults false, so this threshold is
            # measured in frame_id. Moving frame_id from the IMU body to
            # base_footprint drops the origin by 0.192 m (the SDF puts imu_link at
            # base_footprint + (0.338, 0, 0.192)), so the caller must add that back
            # to keep the measured behaviour. See ground_height.
            'Grid/MaxGroundHeight': ground_height,
            'Grid/MaxObstacleHeight': '2.0',

            # The floor yields NO stereo points (blank, untextured), so nothing
            # is ever labelled ground and the grid would carry obstacles with no
            # free space at all. Ray tracing marks the cells between the sensor
            # and each occupied cell as free, which is what makes the grid a map
            # rather than a scatter of hits.
            'Grid/RayTracing': 'true',

            # Off by default. Drops isolated speckle that survives matching.
            'Grid/NoiseFilteringRadius': '0.1',
            'Grid/NoiseFilteringMinNeighbors': '5',

            # How often a keyframe (graph node) is created. This is not just a
            # cost/detail dial -- it sets how many times each map cell is ever
            # OBSERVED, and therefore whether GridGlobal/OccupancyThr has anything
            # to integrate. Measured at 1.0 Hz on dataset_static_tiny_repeat:
            # 26 nodes, and a median occupied cell was inside range AND inside the
            # 90 deg FOV of only 3 keyframes. Demanding 3 net hits then requires
            # perfect detection every single time. See MEMORY_CLAUDE §6c.
            'Rtabmap/DetectionRate': detection_rate,
            'Mem/IncrementalMemory': 'true',

            # --- keep the map level ---------------------------------------
            # Optimizer/GravitySigma is 0.3 by default, meaning "optimise while
            # PRESERVING gravity orientation" -- but it can only do that if the
            # graph actually has gravity links, and it has none unless RTAB-Map
            # gets an orientation reference. The IMU input is remapped away
            # (above), so the only remaining source is the odometry itself, and
            # that is exactly what Mem/UseOdomGravity is for: "use odometry
            # instead of IMU orientation to add gravity links... we assume that
            # odometry is already aligned with gravity (e.g. VIO)". VINS is
            # stereo-INERTIAL and gravity-aligns during IMU init, so its roll and
            # pitch are absolute and safe to trust here.
            #
            # Left false (the upstream default), the graph carried zero Gravity
            # links and loop-closure optimisation was free to rotate the whole
            # map in roll/pitch -- which is what tilted the occupancy grid out of
            # the horizontal plane and away from the trajectory.
            'Mem/UseOdomGravity': 'true',

            # Reject a loop closure whose optimisation error ratio exceeds this.
            # It is the guard that decides whether a DRIFTING front-end can close
            # a loop at all, and the default 3.0 is tuned for odometry far better
            # than a stereo VIO over a warehouse lap. Measured on
            # dataset_static_tiny_repeat: with ground-truth odometry, 72 closures
            # accepted; with VINS, 36 detected by appearance and ALL rejected here
            # ("Rtabmap.cpp:4033 Rejecting all added loop closures"), giving a map
            # with zero closures. Upstream's own multisession demo raises this to
            # 10 for the same reason.
            'RGBD/OptimizeMaxError': optimize_max_error,

            # ---- what the localiser is allowed to ACCEPT ---------------------
            # A warehouse is rows of near-identical shelf bays, the worst case
            # for perceptual aliasing. Measured 11-12 Nov 2026 over four runs: a
            # wrong match is accepted with a LOWER covariance than the honest
            # ones (0.0048 against 0.106), so covariance cannot screen it, and
            # RGBD/OptimizeMaxError cannot either -- it judges consistency with
            # the graph, and a lookalike bay is perfectly consistent, just wrong.
            # Tightening it from 10.0 to 3.0 did not stop the warps and pushed
            # the startup transient from 27 s to 68 s, because the same test also
            # rejects the first CORRECT fix for longer.
            #
            # These two test something OptimizeMaxError does not: how good the
            # geometric match itself is, and whether its inliers are spread
            # across the image or bunched on one piece of repeating structure.
            # Both are at RTAB-Map's own defaults here, so nothing changes until
            # they are passed; see their launch arguments for the values worth
            # trying.
            'Vis/MinInliers': min_inliers,
            'Vis/MinInliersDistribution': min_inliers_distribution,
            # Anti-teleport, localization mode only: verifies a candidate against
            # recent odometry instead of accepting any similar-looking place
            # (Parameters.h:403). Was hard-coded; an argument now so it can be
            # swept with the two above.
            'RGBD/MaxOdomCacheSize': odom_cache_size,

            # 3D->3D instead of RTAB-Map's 3D->2D default. PnP only needs the
            # matched map points to REPROJECT consistently, which a wrongly
            # placed match can still satisfy; 3D->3D requires the two sets of
            # feature positions to be congruent in metres, with
            # Vis/InlierDistance as the tolerance. That is a geometric veto on
            # the sparse map itself, and unlike Reg/Strategy 2 it needs no laser
            # scan -- the feature depths are already in the prior.
            'Vis/EstimationType': estimation_type,
            'Vis/InlierDistance': vis_inlier_distance,

            # Turn the depth image into a laser scan for the LIVE frames, so ICP
            # has something to register when geometric_check is on. The prior
            # side needs rtabmap-reprocess -scan_from_depth; the guard above
            # refuses the run otherwise.
            'gen_scan': gen_scan,
            'gen_scan_max_depth': 8.0,
        }] + [_geometric_check_params(geometric_check, icp_correspondence_ratio,
                                      icp_max_rotation),
              _mode_params(mode, read_only, start_pose),
              _loop_closure_params(loop_closure),
              {'initial_pose': initial_pose} if initial_pose else {}],
        remappings=[
            ('left/image_rect', '/stereo/left/image_rect'),
            ('right/image_rect', '/stereo/right/image_rect'),
            ('left/camera_info', '/stereo/left/camera_info'),
            ('right/camera_info', '/stereo/right/camera_info'),
            # RTAB-Map's "imu" input is a RELATIVE topic name, so it silently
            # resolves to the bag's /imu and then blocks waiting for a TF to the
            # Gazebo IMU frame (ackermann_robot_001/...) that nothing publishes.
            # Point it at a dead topic: the front-end already fuses the IMU and
            # gravity-aligns its output, so RTAB-Map has no use for it here.
            ('imu', '/rtabmap/imu_unused'),
        ])

    # Hold RTAB-Map back for the front end's first seconds.
    #
    # Measured on sweep_J1/J2/J3, 12 Nov 2026. The ORB-SLAM3 orientation is
    # unstable for the first ~1 s it publishes: 15-33 yaw steps over 15 deg,
    # some of 90 deg, between consecutive frames, all inside sim 15.1-16.1 s,
    # after which |dyaw| settles to under 0.1 deg. RTAB-Map's first fix lands
    # within about a second of that window and LATCHES a map->odom correction
    # 73-76 deg wrong, which it then holds for ~16 s until a good localisation
    # replaces it wholesale. J1 and J2 produced 73.8 and 73.7 deg -- the same
    # defect to a tenth of a degree, so this is deterministic, not aliasing.
    #
    # wait_for_imu_init does not stop it: that gate tests Frame::HasVelocity(),
    # which is already true through the unstable window. init_ignore_s does not
    # either -- it lives in the pose gate and only withholds the pose from Nav2,
    # while RTAB-Map has already latched and RViz already shows it wrong.
    #
    # Delaying the node is the blunt version of the right fix and needs no
    # rebuild. The proper one is an orientation-stability condition in
    # slam_wrapper.cpp beside the HasVelocity() test: hold frames back until the
    # last k have turned less than a few degrees each.
    if localize_delay_s > 0.0:
        from launch.actions import TimerAction
        rtabmap = TimerAction(period=localize_delay_s, actions=[rtabmap])
        print(f'[rtab_sim] localise = held back {localize_delay_s:g} s so the first fix '
              f'cannot latch on an unstable front-end orientation')

    nodes = [shim, rtabmap]

    if publish_base_tf:
        nodes.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='base_link_tf', output='screen',
            arguments=['--x', '0', '--y', f'{-imu_height:g}', '--z', '0',
                       '--roll', '0', '--pitch', '1.5707963', '--yaw', '1.5707963',
                       '--frame-id', body_frame, '--child-frame-id', base_frame]))

    # Only started when an output_path is given, so a quick exploratory run does
    # not silently overwrite a recorded result.
    if output_path:
        nodes.append(ExecuteProcess(
            cmd=['python3', WRITER,
                 '--ros-args',
                 '-p', 'use_sim_time:=true',
                 '-p', f'output_path:={output_path}',
                 # RTAB-Map's outputs are RELATIVE topic names on a node with no
                 # namespace, so the optimised graph lands at /mapPath, NOT
                 # /rtabmap/mapPath.
                 '-p', 'path_topic:=/mapPath'],
            output='screen'))

    # Stock RViz, no RTAB-Map plugins: /map, /cloud_map and /mapPath are plain
    # OccupancyGrid / PointCloud2 / Path, so rviz_default_plugins renders them.
    # rtabmap_viz and rtabmap_rviz_plugins are NOT built (see MEMORY_CLAUDE) and
    # are not needed for this.
    if use_rviz:
        nodes.append(Node(
            package='rviz2', executable='rviz2', name='rtab_rviz',
            arguments=['-d', RVIZ_CFG],
            parameters=[{'use_sim_time': True}],
            output='screen'))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'mode', default_value='construct',
            description='construct = build a new map (-d, starts empty). '
                        'extend = open an existing map and add a session, '
                        'relocalizing into it. localize = read-only against an '
                        'existing map, publishing map->odom and '
                        '/localization_pose. construct reproduces the '
                        'pre-mode behaviour exactly.'),
        DeclareLaunchArgument(
            'read_only', default_value='false',
            description='mode:=localize only. Opens the database read-only so the '
                        'prebuilt map provably cannot change (md5sum it either '
                        'side). Incompatible with a database whose sessions never '
                        'merged.'),
        DeclareLaunchArgument(
            'base_frame', default_value='',
            description='Robot frame RTAB-Map reports poses in; empty means use '
                        'body_frame (the pre-mode behaviour). Set base_footprint '
                        'when feeding nav2, so localization_pose and the costmaps '
                        'agree on where the robot is.'),
        DeclareLaunchArgument(
            'optimize_max_error', default_value='3.0',
            description='Reject loop closures whose optimisation error ratio '
                        'exceeds this (0=disabled). 3.0 is the upstream default '
                        'and rejects EVERY closure a drifting VIO front-end '
                        'proposes; upstream multisession uses 10. Raising it '
                        'accepts larger corrections -- and larger mistakes.'),
        DeclareLaunchArgument(
            'loop_closure', default_value='true',
            description='false disables RTAB-Map loop closure detection '
                        '(Rtabmap/LoopThr 1.0, RGBD/ProximityBySpace false) so '
                        'the graph is a pure odometry chain. Visual words are '
                        'still extracted, so the database stays usable for '
                        'mode:=localize. Pair with loopClosing: 0 in the ORB '
                        'config for a map that shows only odometry drift.'),
        DeclareLaunchArgument(
            'ground_height', default_value='0.1',
            description='Grid/MaxGroundHeight, measured in base_frame because '
                        'Grid/MapFrameProjection is false. If base_frame is '
                        'base_footprint rather than the IMU body, add 0.192 (the '
                        'SDF offset) to keep the measured behaviour: 0.29.'),
        DeclareLaunchArgument(
            'frontend', default_value='',
            description='orb | vins: sets odom_frame/body_frame to what that '
                        'front end broadcasts (ORB: orbslam3_world -> imu_link on '
                        'the fisheye rig, -> base_footprint in simulation; VINS: '
                        'world -> body). Empty keeps the explicit frame arguments.'),
        DeclareLaunchArgument(
            'publish_base_tf', default_value='auto',
            description='Publish the FLU base_link under body_frame from this '
                        'launch (no separate static_transform_publisher). auto = '
                        'only for camera_model:=fisheye, whose IMU body frame is '
                        'x-right/y-up/z-back; false to run your own.'),
        DeclareLaunchArgument(
            'imu_height', default_value='0.11',
            description='Height of the IMU body frame above the floor [m], used '
                        'for the base_link transform (measured 0.11 m, 4 Nov 2026).'),
        DeclareLaunchArgument(
            'odom_frame', default_value='world',
            description='Odometry frame the front-end broadcasts: "world" for '
                        'VINS (world_frame_id), "orbslam3_world" for ORB-SLAM3.'),
        DeclareLaunchArgument(
            'body_frame', default_value='body',
            description='Frame the odometry resolves to: "body" for VINS, '
                        '"base_footprint" for ORB-SLAM3.'),
        DeclareLaunchArgument(
            'output_path', default_value='',
            description='Directory for vio.csv (a path ending in .csv is used as '
                        'given), e.g. output/output_vins_rtab/simulation/'
                        'dataset_dynamic_000. Holds RTAB-Map\'s OPTIMISED graph, '
                        'one pose per node at Rtabmap/DetectionRate -- far sparser '
                        'than the front-ends\' per-frame vio.csv. Empty means do '
                        'not write one.'),
        DeclareLaunchArgument(
            'database_path', default_value='~/output/rtabmap/wil_sim.db',
            description='RTAB-Map database FILE, ending in .db. Anything else is '
                        'treated as a directory and rtabmap.db is created inside '
                        'it -- because the node runs with -d, which ERASES this '
                        'path at startup. Never point it at a results directory. '
                        'Grows with every keyframe; watch disk.'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Open RViz on /map, /cloud_map and /mapPath. RTAB-Map only '
                        'assembles those when something subscribes, so leaving this '
                        'false costs nothing.'),
        DeclareLaunchArgument(
            'shim_rate', default_value='4.0',
            description='Hz the shim republishes stereo pairs at. This is a HARD '
                        'CAP on detection_rate -- RTAB-Map cannot make a keyframe '
                        'from a frame it never received, so detection_rate above '
                        'this value silently does nothing. Raise both together.'),
        DeclareLaunchArgument(
            'detection_rate', default_value='1.0',
            description='Keyframe rate (Hz). Raising it gives each map cell more '
                        'observations, which is what a multi-hit occupancy gate '
                        'needs. Capped by the shim republish rate (4 Hz). Costs '
                        'database size and vio.csv rows.'),
        DeclareLaunchArgument(
            'grid_range_max', default_value='10.0',
            description='Metres beyond which stereo points are discarded. '
                        'Swept against a baked ground-truth map: recall rises '
                        '45 -> 51 % from 4 m to 10 m while precision only falls '
                        '94 -> 89 %, so the far points are mostly real. Was 4.0, '
                        'chosen to crop smears that turned out to be a camera-TF '
                        'bug (see MEMORY_CLAUDE section 0, fix 13).'),
        DeclareLaunchArgument(
            'wait_for_transform', default_value='5.0',
            description='Seconds to wait for the front-end odometry TF covering '
                        'an image stamp. Raise if the front-end lags playback; '
                        'the real fix is a slower --rate so its backlog drains.'),
        DeclareLaunchArgument(
            'map_always_update', default_value='false',
            description='Keep the assembled-map cache even with no subscribers. '
                        'Measured unnecessary for a late-attaching RViz, which '
                        'still gets the full map; costs CPU/memory every run.'),
        DeclareLaunchArgument(
            'delete_db', default_value='',
            description="Empty (default) means the mode decides: only construct "
                        "wipes. true is REFUSED on extend/localize rather than "
                        "honoured, because -d would erase the map they exist to "
                        "read. false forces append even in construct."),
        DeclareLaunchArgument(
            'config_dir', default_value=CONFIG_DIR,
            description='VINS config directory the shim reads its calibration '
                        'from. pinhole: cam{0,1}_pinhole.yaml + stereo_imu.yaml '
                        '(sim). fisheye: cam{0,1}_fisheye.yaml + stereo_imu.yaml '
                        '(real rig: vins_fusion_ros2/config/wil).'),
        DeclareLaunchArgument(
            'min_inliers', default_value='20',
            description='Vis/MinInliers: feature correspondences a match must survive '
                        'to be accepted. RTAB-Map default 20; try 35 when lookalike '
                        'shelf bays are being accepted.'),
        DeclareLaunchArgument(
            'min_inliers_distribution', default_value='0.0',
            description='Vis/MinInliersDistribution: how spread the inliers must be '
                        'across the image (second PCA eigenvalue, 0 to 0.5). 0 disables '
                        'it, which is the RTAB-Map default. Try 0.05-0.1: it is the one '
                        'test aimed at a match bunched on one piece of repeating '
                        'structure, which is what an aisle warp is.'),
        DeclareLaunchArgument(
            'odom_cache_size', default_value='10',
            description='RGBD/MaxOdomCacheSize: odometry poses kept to verify a '
                        'localisation against recent motion. Localize mode only. '
                        'Larger gives the anti-teleport check more context.'),
        DeclareLaunchArgument(
            'geometric_check', default_value='false',
            description='Reg/Strategy 2: verify every accepted loop closure by ICP '
                        'against the 3-D structure, not only the 2-D features that '
                        'proposed it. RTAB-Map default is Vis-only, which consults no '
                        'geometry at all. This is the one test that is outside the null '
                        'space defeating OptimizeMaxError and MaxOdomCacheSize; it costs '
                        'ICP on the depth cloud per candidate.'),
        DeclareLaunchArgument(
            'icp_correspondence_ratio', default_value='0.3',
            description='Icp/CorrespondenceRatio, the actual veto. RTAB-Map default 0.1; '
                        'raised because a 73 deg misplacement should not find a third of '
                        'its points. geometric_check:=true only.'),
        DeclareLaunchArgument(
            'icp_max_rotation', default_value='0.35',
            description='Icp/MaxRotation (rad), cap on how far ICP may rotate a closure '
                        'to make it fit. RTAB-Map default 0.78 (45 deg) is wide enough to '
                        'absorb a 23-24 deg error; 0.35 is 20 deg. geometric_check:=true '
                        'only.'),
        DeclareLaunchArgument(
            'localize_delay_s', default_value='0.0',
            description="seconds to hold RTAB-Map back after the front end starts, so "
                        "its first fix cannot latch on the unstable orientation of the "
                        "first ~1 s. 0 disables. Unlike init_ignore_s, which only "
                        "withholds the pose from Nav2, this stops the wrong correction "
                        "being formed at all."),
        DeclareLaunchArgument(
            'estimation_type', default_value='1',
            description='Vis/EstimationType: 1 = 3D->2D PnP (RTAB-Map default), '
                        '0 = 3D->3D, which requires the matched feature positions to '
                        'agree in metres rather than merely reproject consistently. 0 is '
                        'the sparse-geometry veto and needs no laser scan.'),
        DeclareLaunchArgument(
            'vis_inlier_distance', default_value='0.1',
            description='Vis/InlierDistance (m), the tolerance of the 3D->3D check. '
                        'estimation_type:=0 only.'),
        DeclareLaunchArgument(
            'gen_scan', default_value='false',
            description='Generate a laser scan from the depth image for the live frames. '
                        'Needed by geometric_check:=true, whose ICP registers scans and '
                        'not depth. The prior needs its own, via '
                        'rtabmap-reprocess -scan_from_depth.'),
        DeclareLaunchArgument(
            'relocalize', default_value='true',
            description='localize mode only. false = dead reckon on the prior: the pose '
                        'is the start_pose seed composed with the front end odometry, the '
                        "prior's grid is still served, and no correction is ever applied. "
                        'Drift is then unbounded and the pose gate cannot detect it.'),
        DeclareLaunchArgument(
            'start_pose', default_value='origin',
            description="localize mode only. 'origin' assumes the robot restarts "
                        "at the map's first node, which on these priors is the "
                        "spawn pose to within 5 cm; 'last' restarts from the pose "
                        'left in the database, which with read_only is where the '
                        'MAPPING run ended; or six numbers "x y z roll pitch yaw" '
                        'to seed an explicit pose in the map frame.'),
        DeclareLaunchArgument(
            'camera_model', default_value='pinhole',
            description='pinhole = sim bags, raw frames are already rectified. '
                        'fisheye = real rig; the shim remaps every pair onto a '
                        'virtual rectified pinhole pair (hfov_deg) because '
                        'RTAB-Map has no equidistant stereo model.'),
        DeclareLaunchArgument(
            'left_cam', default_value='0',
            description='fisheye only: index of the physically LEFT camera. '
                        'Validated against the extrinsic at startup (the shim '
                        'exits if that camera is on the right). Since the '
                        '2026-10-19 remount cam1 is left on the real rig: left_cam:=1.'),
        DeclareLaunchArgument(
            'rect_scale', default_value='1.0',
            description='fisheye only: rectified output size as a fraction of '
                        'the 1920x1080 input. 0.5 cuts RTAB-Map dense stereo '
                        '(SGBM) cost ~4-8x; use it when the back-end cannot keep '
                        'up. Depth precision per pixel halves with it.'),
        DeclareLaunchArgument(
            'hfov_deg', default_value='110',
            description='fisheye only: horizontal FOV of the virtual pinhole. '
                        '110 reproduces the sim depth budget (fx*B ~ 60 px*m); '
                        '130 keeps more of the fisheye at 2/3 the depth precision.'),
        OpaqueFunction(function=launch_setup),
    ])
