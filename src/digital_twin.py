import sys
sys.path.append("/home/fhagelskjaer/workspace/i40-binpicking/DigitalTwin/")
from DigitalTwin import *
from scipy.spatial.transform import Rotation as R
import numpy as np
from TB.src.pose_estimation import *
from TB.src.main import *
import TB.src.main
import robotiq_gripper

base2cam = np.array([
            [-0.36984668310865088, -0.92906206999578722, -0.0075565262305073745, 0.1786705608504808],
            [ -0.928553085539913, 0.36989679530527891, -0.031072948961943965, -0.3897976193245112],
            [ 0.031663833119762057, -0.004475591360678462, -0.99948855458886376, 1.109239730618373],
            [ 0, 0, 0, 1]] )


if __name__ == "__main__":

    print("Connecting to Robot")
    rtde_c = rtde_control.RTDEControlInterface("172.28.60.10")
    rtde_r = rtde_receive.RTDEReceiveInterface("172.28.60.10")

    # CONNECT GRIPPER
    print("Creating gripper...")
    gripper = robotiq_gripper.RobotiqGripper()
    print("Connecting to gripper...")
    gripper.connect("172.28.60.10", 63352)
    #print("Activating gripper...")
    #gripper.activate()
    gripper.move_and_wait_for_pos(0, 100, 100)

    wc = loader.WorkCellLoaderFactory.load(
        "/home/fhagelskjaer/workspace/i40-binpicking/DigitalTwin/WorkCell/WorkCellClip.wc.xml")

    print("Rtde is connected:", rtde_r.isConnected())
    twin = DigitalTwin(wc, rtde_c, rtde_r)
    twin.showTwin()


    index = 0
    while True:
        print(index)
        dir_path = "/home/fhagelskjaer/workspace/i40-binpicking/DigitalTwin/Clip_" + str(index) + "/"
        try:
            os.mkdir(dir_path)
        except:
            pass
        f = open(dir_path + 'log.txt', "a")

        home_pose = [-0.3813916166221362, -0.35049373973032844, 0.5124741717261786, -3.1017059579089814, 0.4988549930615818, 3.422761748430787e-05]
        bin_pose = [-0.5833302555027325, -0.0039042374721320414, 0.39885648665915313, 2.640770982345075, -1.629908950908474, -0.044196917691981044]

        arrowframe = twin.getFrames("arrow")[0]
        twin.moveFrame(arrowframe, home_pose)
        valid, path = twin.pathTo(Pose=home_pose)
        print("Path for Home Pose is valid: ", valid)
        if valid:
            for i in range(len(path)):
                rtde_c.moveJ(QasList((path[i])), 2, 2)
        else:
            continue

        
        input_query = input("...\n")
        success, pick_transform, place_transform, log = full_pipeline(use_camera=True, query=input_query, save_output=True, index=i)
        if not success:
            print("Pipeline failed")
            f.write(str(log))
            index += 1
            continue
        pick_transform[0:3,3] /= 1000
        print("Pick Success:{} from \n{} \nto \n{}".format(success, pick_transform, place_transform))
        

        
        cam2obj = pick_transform
        # cam2obj = np.eye(4)


        obj2tcp = np.array(
            [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -.38],
            [0, 0, 0, 1]]
        )
        
        base2tcp = base2cam @ cam2obj @ obj2tcp
    
        print("final pose: \n{}".format(base2tcp))

        r = R.from_matrix(base2tcp[:3, :3])
        inspect_pose = [base2tcp[0, 3], base2tcp[1, 3], base2tcp[2, 3], r.as_rotvec()[0],
                        r.as_rotvec()[1], r.as_rotvec()[2]]
        
        # inspect_pose = [0,0,0, 0,0,0]
        twin.moveFrame(arrowframe, inspect_pose)
    
        
        
    
        valid, path = twin.pathTo(Pose=inspect_pose)
        
        log['path to pick'] = valid
        
        print("Path for pick is Valid: ", valid)
        if valid:
            log ['path to pick'] = True
            for i in range(len(path)):
                rtde_c.moveJ(QasList((path[i])), 2, 2)
        else:
            f.write(str(log))
            index += 1
            continue

        tcp_grasp_position = inspect_pose
        tcp_grasp_position[2] -= 0.10
        rtde_c.moveL(tcp_grasp_position, 0.075, 1.0, asynchronous=True)

        contact_detected = False
        ts = time.time()
        while True:
            if rtde_c.toolContact([0, 0, 0, 0, 0, 0]) > 0 and time.time() - ts > 0.10:
                rtde_c.stopL(5.0)
                contact_detected = True
                print("Contact detected")
                break
            time.sleep(0.002)

        # Grip
        gripper.move_and_wait_for_pos(255, 100, 5)
        
        #Go up again

        tcp_grasp_position = inspect_pose
        tcp_grasp_position[2] += 0.20
        rtde_c.moveL(tcp_grasp_position, 0.075, 1.0, asynchronous=True)

        #Go to place
        valid, path = twin.pathTo(Pose=bin_pose)
        log['path to place'] = valid
        print("Path for inspect (1) is Valid: ", valid)
        if valid:
            for i in range(len(path)):
                rtde_c.moveJ(QasList((path[i])), 2, 2)
        else:
            f.write(str(log))
            index += 1
            continue

        # Grip
        gripper.move_and_wait_for_pos(0, 100, 5)
        
        picking_success = input("success input?\n")
        log['picking success'] = picking_success
        objs_present = input("What objs?\n")
        log['objs present'] = objs_present

        f.write(str(log))
        index += 1

    while twin.showRunning():
        time.sleep(1)
        
        
