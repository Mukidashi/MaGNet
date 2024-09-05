import os
import numpy as np 
from PIL import Image
import cv2
import yaml
import torch


def read_pose_file(pose_file):
    fp = open(pose_file,'r')
    
    fstr = next(fp,None)

    pose = np.zeros((3,4)) 
    for i in range(3):
        fstr = next(fp,None)

        felems = fstr.strip().split()
        pose[i][0] = float(felems[0])
        pose[i][1] = float(felems[1])
        pose[i][2] = float(felems[2])
        pose[i][3] = float(felems[3])

    fp.close()

    return pose


def yaml_construct_opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node,deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


def read_calib(calib_path):
    yfp = open(calib_path,"r")
    next(yfp,None)
    yaml.add_constructor(u'tag:yaml.org,2002:opencv-matrix', yaml_construct_opencv_matrix,yaml.SafeLoader)
    calib = yaml.safe_load(yfp)
    yfp.close()

    return calib


class RoadVidData:
    def __init__(self, opts):
        self.pose_dir = opts.pose_dir
        self.img_dir = opts.img_dir
        self.device = 'cuda'

        self.load_datas()


        calib = read_calib(opts.calib_yaml)

        self.calib_wid = calib["Camera.width"]
        self.calib_hei = calib["Camera.height"]

        self.Kmat = np.eye(3)
        self.Kmat[0,0] = calib["Camera.fx"]
        self.Kmat[1,1] = calib["Camera.fy"]
        self.Kmat[0,2] = calib["Camera.cx"]
        self.Kmat[1,2] = calib["Camera.cy"]

        self.width = opts.input_width
        self.height = opts.input_height
        self.dpv_W = opts.dpv_width
        self.dpv_H = opts.dpv_height

        self.setup_intrinsics()
        



    def __len__(self):
        return len(self.poseList) - 1


    def load_datas(self):

        fp = open(os.path.join(self.pose_dir,'pose.list'),'r')

        self.imgPathList = []
        self.poseList = []

        bFirst = True
        for line in fp:
            lelems = line.strip().split()
            if len(lelems) < 3:
                continue
            
            if bFirst:
                self.imgPathList.append(os.path.join(self.img_dir,"{0}.jpg".format(lelems[1])))
                bFirst = False

            self.imgPathList.append(os.path.join(self.img_dir,"{0}.jpg".format(lelems[0])))

            pose = read_pose_file(os.path.join(self.pose_dir,lelems[2]))
            self.poseList.append(pose)

        fp.close()


    def get_data(self,idx):
        
        if idx < 0 or idx >= len(self.poseList)-1:
            print("Index:{0} is out of range(0-{1})".format(idx,len(self.poseList)-1))
            return None

        ## Read images (resize, crop, adjust Kmat)
        ref_img = cv2.imread(self.imgPathList[idx+1])
        ref_rgb = self.process_image(ref_img)
        ref_rgb = torch.from_numpy(ref_rgb).permute(2,0,1)
        ref_rgb = ref_rgb.to(self.device)

        img_list = [] 
        img0 = cv2.imread(self.imgPathList[idx])
        rgb0 = self.process_image(img0)
        rgb0 = torch.from_numpy(rgb0).permute(2,0,1)
        rgb0 = rgb0.to(self.device)
        img_list.append(rgb0.unsqueeze(0))

        img1 = cv2.imread(self.imgPathList[idx+2])
        rgb1 = self.process_image(img1)
        rgb1 = torch.from_numpy(rgb1).permute(2,0,1)
        rgb1 = rgb1.to(self.device)
        img_list.append(rgb1.unsqueeze(0))
        
        nghbr_imgs = torch.cat(img_list,dim=0) 


        ## Set pose mat (world to camera)
        # ref_pose = np.eye(4)
        # ref_pose = torch.from_numpy(ref_pose).float().to(self.device)

        nghbr_poses = torch.zeros((1,2,4,4))
        is_valid = torch.ones((1,2),dtype=torch.int)

        # pose_list = []
        pose_0r = np.eye(4)
        pose_0r[:3,:] = self.poseList[idx]
        nghbr_poses[0][0] = torch.from_numpy(pose_0r.astype(np.float32)).float()
        # pose_0r = torch.from_numpy(pose_0r.astype(np.float32)).float().to(self.device)
        # pose_list.append(pose_0r.unsqueeze(0))

        pose_1r = np.eye(4)
        for i in range(3):
            pose_1r[i][3] = 0.0
            for j in range(3):
                pose_1r[i][j] = self.poseList[idx+1][j][i]
                pose_1r[i][3] -= self.poseList[idx+1][j][i]*self.poseList[idx+1][j][3]
        nghbr_poses[0][1] = torch.from_numpy(pose_1r.astype(np.float32)).float()
        # pose_1r = torch.from_numpy(pose_1r.astype(np.float32)).float().to(self.device)
        # pose_list.append(pose_1r.unsqueeze(0))


        ## Kmat
        # K_pool = {}
        # for i in range(6):
        #     K_pool[(self.height//2**i,self.width//2**i)] = self.scaleKmat.copy().astype('float32')
        #     K_pool[(self.height//2**i,self.width//2**i)][:2,:] /= 2**i
        
        # inv_K_pool = {}
        # for k, v in K_pool.items():
        #     K44 = np.eye(4)
        #     K44[:3,:3] = v
        #     invK = np.linalg.inv(K44).astype(np.float32)
        #     inv_K_pool[k] = torch.from_numpy(invK).to(self.device).unsqueeze(0)
        pix_to_ray = self.pix_to_ray.unsqueeze(0)
        intM = torch.from_numpy(self.dpvKmat.astype(np.float32))
        cam_intrinsics = {
            'unit_ray_array_2D': pix_to_ray,
            'intM': intM.unsqueeze(0)
        }

        return ref_rgb.unsqueeze(0),nghbr_imgs.to(self.device),nghbr_poses.to(self.device), is_valid, cam_intrinsics


    def process_image(self,img):
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img = cv2.resize(img,(self.img_resize[0],self.img_resize[1]),interpolation=cv2.INTER_LINEAR)
        
        # bgr = np.zeros((self.height,self.width,3),dtype=np.uint8)
        # bgr[self.topleft[1]:self.topleft[1]+self.img_resize[1],self.topleft[0]:self.topleft[0]+self.img_resize[0],:] = img
        bgr = img[self.topleft[1]:self.topleft[1]+self.height,self.topleft[0]:self.topleft[0]+self.width,:]
        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

        rgb = rgb.astype(np.float32)/255.0
        rgb = (rgb-mean)/std

        return rgb.astype(np.float32)
   

    def setup_intrinsics(self):
        scale_x = float(self.width)/float(self.calib_wid)
        scale_y = float(self.height)/float(self.calib_hei)

        self.img_resize = [self.width,self.height]
        self.topleft = [0,0]
        self.img_scale = 1.0
        # if scale_x <= scale_y:
        #     self.img_resize[1] = int(self.calib_hei*scale_x)
        #     self.topleft[1] = int((self.height - self.img_resize[1])/2)
        #     self.img_scale = scale_x
        # else:
        #     self.img_resize[0] = int(self.calib_wid*scale_y)
        #     self.topleft[0] = int((self.width - self.img_resize[0])/2)
        #     self.img_scale = scale_y
        if scale_x <= scale_y:
            self.img_resize[0] = int(self.calib_wid*scale_y)
            self.topleft[0] = int((self.img_resize[0]-self.width)/2)
            self.img_scale = scale_y
        else:
            self.img_resize[1] = int(self.calib_hei*scale_x)
            self.topleft[1] = int(( self.img_resize[1] - self.height)/2)
            self.img_scale = scale_x

        print(self.img_resize)
        print(self.topleft)
        print(self.img_scale)

        self.scaleKmat = np.eye(3)
        self.scaleKmat[0,0] = self.Kmat[0,0]*self.img_scale
        self.scaleKmat[1,1] = self.Kmat[1,1]*self.img_scale
        # self.scaleKmat[0,2] = self.Kmat[0,2]*self.img_scale + float(self.topleft[0])
        # self.scaleKmat[1,2] = self.Kmat[1,2]*self.img_scale + float(self.topleft[1])
        self.scaleKmat[0,2] = self.Kmat[0,2]*self.img_scale - float(self.topleft[0])
        self.scaleKmat[1,2] = self.Kmat[1,2]*self.img_scale - float(self.topleft[1])

        self.dpvKmat = np.eye(3)
        self.dpvKmat[0][0] = self.scaleKmat[0][0]*(float(self.dpv_W)/float(self.width))
        self.dpvKmat[1][1] = self.scaleKmat[1][1]*(float(self.dpv_H)/float(self.height))
        self.dpvKmat[0][2] = self.scaleKmat[0][2]*(float(self.dpv_W)/float(self.width))
        self.dpvKmat[1][2] = self.scaleKmat[1][2]*(float(self.dpv_H)/float(self.height))

        print(self.scaleKmat, self.dpvKmat)

        ray_array = np.ones((self.dpv_H,self.dpv_W,3))
        x_range = np.arange(self.dpv_W)
        y_range = np.arange(self.dpv_H)
        x_range = np.concatenate([x_range.reshape(1,self.dpv_W)]*self.dpv_H,axis=0)
        y_range = np.concatenate([y_range.reshape(self.dpv_H,1)]*self.dpv_W,axis=1)
        ray_array[:,:,0] = x_range + 0.5
        ray_array[:,:,1] = y_range + 0.5

        pixel_to_ray_array = np.copy(ray_array)
        pixel_to_ray_array[:,:,0] = ((pixel_to_ray_array[:,:,0]*(self.width/float(self.dpv_W)))-self.scaleKmat[0,2])/self.scaleKmat[0,0]
        pixel_to_ray_array[:,:,1] = ((pixel_to_ray_array[:,:,1]*(self.height/float(self.dpv_H)))-self.scaleKmat[1,2])/self.scaleKmat[1,1]

        pixel_to_ray_array_2D = np.reshape(np.transpose(pixel_to_ray_array,axes=[2,0,1]),[3,-1])
        self.pix_to_ray = torch.from_numpy(pixel_to_ray_array_2D.astype(np.float32))

    def postprocess_depth(self,depth):

        return depth
        
