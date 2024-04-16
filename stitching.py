
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree


import random

def Distance(img1,img2,img1_xmin,img1_xmax,img1_ymin,img2pose_x,img2pose_y,img2orjx,img2orjy):
    # leftdist
    #
    # a,b = img1.shape
    # c,d = img2.shape
    # img1_center = [int(a/2),int(b/2)]
    # img2_center = [int(c/2),int(d/2)]
    # Inversed_img2_center =
    # Inversed_img2_center=np.dot(inversehomography,np.transpose(img2left_upc))
    # Inversed_img2_center = (1/img2_leftupcorr.item(2))*img2_leftupcorr


    return alpha

def EucDistanceFun(centerx, centery, currx, curry):
    distance = np.sqrt(np.square(centerx - currx) + np.square(centery - curry))

    return distance

def DynamicEucledean_Interpoltion(canvas,centerx,centery,Eucdistance=4):

    channel_values=np.zeros((3,1))
    patch_lupcoor_x,patch_lupcoor_y, patch_rdowncoor_x,patch_rdowncoor_y=(centerx-Eucdistance,centery-Eucdistance,centerx+Eucdistance,centery+Eucdistance)
    #this will use for total channel values for each pixel in the spatial of center pixel max(channel value)>0 than calculate it
    totalrgbvalue=0
    #this will use for normalization of totalrgbvalues take the weights of each pixel by looking euclidean distance and sum them up
    normalization_value=-1


    #in this way we dont look at borders of canvas so our algorithm work correctly by looking neigbour pixels
    if(patch_lupcoor_x>=0 and patch_lupcoor_y>=0 and patch_rdowncoor_x<(len(canvas[0])) and patch_rdowncoor_y<(len(canvas)-1)):
        if(max(canvas[centery][centerx-1])>0 and max(canvas[centery][centerx+1])>0 or max(canvas[centery-1][centerx])>0 and   max(canvas[centery+1][centerx])>0   ):
                for y in range (patch_lupcoor_y,patch_rdowncoor_y):
                    for x in range (patch_lupcoor_x,patch_rdowncoor_x):
                        if(max(canvas[y][x])>0):
                            totalrgbvalue+=np.multiply((1/EucDistanceFun(centerx,centery,x,y)),canvas[y][x])
                            normalization_value+=(1/EucDistanceFun(centerx,centery,x,y))

                channel_values=totalrgbvalue/normalization_value








    return channel_values


def  Canvas_Intersect(canvas,img1,img2,inversehomography,proj1_leftupc,proj1_rightupc,proj1_leftdownc,proj1_rightdownc,shiftx,shifty):
    proj_canvas=np.zeros((len(canvas),len(canvas[0]),3))


    min_intersectx=len(img1[0])
    max_intersectx=0

    # max_intersect=0
    for y in range(len(img2)):
        for x in range(len(img2[0])):
            proj2 = np.dot(inversehomography, np.transpose([x, y, 1]))
            proj2 = (1 / proj2.item(2)) * proj2

            if(((proj2[0]>=proj1_leftupc[0]) or (proj2[0]>=proj1_leftdownc[0])) and ((proj2[0]<=proj1_rightupc[0])or (proj2[0]<=proj1_rightdownc[0])))   and (((proj2[1]>=proj1_leftupc[1]) or (proj2[1]>=proj1_rightupc[1])) and ((proj2[1]<=proj1_leftdownc[1]) or (proj2[1]<=proj1_rightdownc[1]))):

                if (min_intersectx >( int(proj2[0])+shiftx)):
                    min_intersectx = int(proj2[0])+shiftx

                if (max_intersectx <(int(proj2[0])+shiftx)):
                    max_intersectx = int(proj2[0])+shiftx
    normalization=max_intersectx-min_intersectx
    for y in range(len(img2)):
        for x in range(len(img2[0])):
            proj2=np.dot(inversehomography,np.transpose([x,y,1]))
            proj2 = (1 / proj2.item(2)) * proj2


            if(((proj2[0]>=proj1_leftupc[0]) or (proj2[0]>=proj1_leftdownc[0])) and ((proj2[0]<=proj1_rightupc[0])or (proj2[0]<=proj1_rightdownc[0])))   and (((proj2[1]>=proj1_leftupc[1]) or (proj2[1]>=proj1_rightupc[1])) and ((proj2[1]<=proj1_leftdownc[1]) or (proj2[1]<=proj1_rightdownc[1]))):

                #canvas[int(proj2[1]+shifty)][int(proj2[0])+shiftx]=img2[y][x] #directly project image 2 pixels to intersect with img1 boundries
                #alpha=Distance(img1,img2,img1_xmin,img1_xmax,img1_ymin,img2pose_x,img2pose_y,img2orjx,img2orjy)
                alpha=np.divide(int(proj2[0])+shiftx-min_intersectx,normalization)
                canvas[int(proj2[1])+shifty][int(proj2[0])+shiftx] =alpha*img2[y][x]+(1-alpha)*canvas[int(proj2[1])+shifty][int(proj2[0])+shiftx]
            else:
                canvas[int(proj2[1])+shifty][int(proj2[0])+shiftx]=img2[y][x]
    #Interpolation part
    iterate_interpolation=5
    
    for i in range(iterate_interpolation):
        for y in range(1,len(canvas)-1):
            for x in range(1,len(canvas[0])-1):
                if(max(canvas[y][x])==0):
                    Estimatepixelrgb=DynamicEucledean_Interpoltion(canvas,x,y)
                    canvas[y][x][0]=Estimatepixelrgb[0]
                    canvas[y][x][1] = Estimatepixelrgb[1]
                    canvas[y][x][2] = Estimatepixelrgb[2]






    #blend_img=Alpha_Blend_img(img1,img2,inverseHomography,projCanvas)



    return canvas









def Canvas(img1,img2,inversehomography):
    img2left_upc=[0,0,1]
    img2right_upc=[len(img2[0]),0,1]
    img2left_downc=[0,len(img2),1]
    img2right_downc=[len(img2[0]),len(img2),1]
    #Down part find the correspondance cordinates with respected to the (0,0) cordinate to the image1

    img2_leftupcorr=np.dot(inversehomography,np.transpose(img2left_upc))
    img2_leftupcorr = (1/img2_leftupcorr.item(2))*img2_leftupcorr
    img2_rightupcorr = np.dot(inversehomography,np.transpose(img2right_upc) )
    img2_rightupcorr = (1 / img2_rightupcorr.item(2)) * img2_rightupcorr
    img2_leftdowncorr = np.dot(inversehomography,np.transpose(img2left_downc))
    img2_leftdowncorr =(1 / img2_leftdowncorr.item(2)) * img2_leftdowncorr
    img2_rightdowncorr = np.dot(inversehomography,np.transpose(img2right_downc))
    img2_rightdowncorr =(1 / img2_rightdowncorr.item(2)) * img2_rightdowncorr

    '''
    img2_leftupcorr = np.dot(img2left_upc,inversehomography)
    img2_leftupcorr = (1 / img2_leftupcorr.item(2)) * img2_leftupcorr
    img2_rightupcorr = np.dot(img2right_upc,inversehomography)
    img2_rightupcorr = (1 / img2_rightupcorr.item(2)) * img2_rightupcorr
    img2_leftdowncorr = np.dot(img2left_downc,inversehomography)
    img2_leftdowncorr = (1 / img2_leftdowncorr.item(2)) * img2_leftdowncorr
    img2_rightdowncorr = np.dot(img2right_downc,inversehomography)
    img2_rightdowncorr = (1 / img2_rightdowncorr.item(2)) * img2_rightdowncorr
    '''
    max_x=max(len(img1[0]),img2_rightupcorr.item(0),img2_rightdowncorr.item(0))
    min_x= min(0,img2_leftupcorr.item(0),img2_leftdowncorr.item(0))
    max_y = max(len(img1),img2_leftdowncorr.item(1),img2_rightdowncorr.item(1))
    min_y =min(0, img2_leftupcorr.item(1),img2_rightupcorr.item(1))

    a = int(max_x-min_x)
    b = int(max_y-min_y)
    canvas_img=np.zeros((b,a,3))
    #our new coordinates for image1 on canvas
    proj1_leftupc,proj1_rightupc,proj1_leftdownc,proj1_rightdownc=(0,0,0,0)

    if min_x<0 and min_y<0:
        proj1_leftupc=  [int(np.abs(min_x)),int(abs(min_y))]
        proj1_rightupc= [int(np.abs(min_x))+len(img1[0]),int(abs(min_y))]
        proj1_leftdownc=[int(np.abs(min_x)),int(abs(min_y))+len(img1)]
        proj1_rightdownc=[int(np.abs(min_x))+len(img1[0]),int(abs(min_y))+len(img1)]

        for y in range(len(img1)):
            for x in range(len(img1[0])):
                canvas_img[y+int(abs(min_y))][x+int(np.abs(min_x))]=img1[y][x]
        proj_canvas = Canvas_Intersect(canvas_img, img1, img2, inversehomography, proj1_leftupc, proj1_rightupc,
                                       proj1_leftdownc, proj1_rightdownc,int(np.floor(np.abs(min_x))),int(np.floor(np.abs(min_y))))

    elif min_x < 0 and min_y >= 0:
        proj1_leftupc = [int(np.abs(min_x)),0]
        proj1_rightupc = [int(np.abs(min_x)) + len(img1[0]), 0]
        proj1_leftdownc = [int(np.abs(min_x)),  len(img1)]
        proj1_rightdownc = [int(np.abs(min_x))+ len(img1[0]), len(img1)]
        for y in range(len(img1)):
            for x in range(len(img1[0])):
                canvas_img[y][x + int(np.abs(min_x))] = img1[y][x]
        proj_canvas = Canvas_Intersect(canvas_img, img1, img2, inversehomography, proj1_leftupc, proj1_rightupc,
                                       proj1_leftdownc, proj1_rightdownc,int(np.floor(np.abs(min_x))),0)

    elif min_x >= 0 and min_y < 0:
        proj1_leftupc = [0, int(abs(min_y))]
        proj1_rightupc = [len(img1[0]), int(abs(min_y))]
        proj1_leftdownc = [0, int(abs(min_y)) + len(img1)]
        proj1_rightdownc = [len(img1[0]), int(abs(min_y)) + len(img1)]
        for y in range(len(img1)):
            for x in range(len(img1[0])):
                canvas_img[y+int(np.abs(min_y))][x] = img1[y][x]
        proj_canvas = Canvas_Intersect(canvas_img, img1, img2, inversehomography, proj1_leftupc, proj1_rightupc,
                                       proj1_leftdownc, proj1_rightdownc,0,int(np.floor(np.abs(min_y))))

    else:
        proj1_leftupc = [0,0]
        proj1_rightupc = [len(img1[0]), 0]
        proj1_leftdownc = [0, len(img1)]
        proj1_rightdownc = [len(img1[0]),len(img1)]
        for y in range(len(img1)):
            for x in range(len(img1[0])):
                canvas_img[y][x] = img1[y][x]
        proj_canvas = Canvas_Intersect(canvas_img, img1, img2, inversehomography, proj1_leftupc, proj1_rightupc,
                                       proj1_leftdownc, proj1_rightdownc,0,0)


    return canvas_img,proj_canvas


def homography(Corresponding_points):
    # print("Corresponding_points")
    # print(Corresponding_points)
    correspondenceList=[]
    count=0
    x1,y1,x2,y2=0,0,0,0

    for match in Corresponding_points:
        '''
        if(count%2==0):
            (x1, y1) = (match.item(0), match.item(1))
        else:
            (x2, y2) = (match.item(0), match.item(1))
            correspondenceList.append([x1, y1, x2, y2])
        '''


        ''' We need to obtain x1 and y1 coordianates from matchhes '''
        (x1, y1) = match[0][match.trainIdx].pt
        (x2, y2) = match[1][match.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])

    h=np.zeros(3,3)
    A=np.zeros(len(correspondenceList),9)
    for x in range (len(A)/2):
        A[x*2]=[correspondenceList[x*2][0], correspondenceList[x*2][1],1,0,0,0,-correspondenceList[x*2][2]*correspondenceList[x*2][0],-correspondenceList[x*2][2]*correspondenceList[x*2][1],-correspondenceList[x*2][2]]
        A[x*2+1]=[0,0,0,correspondenceList[x*2][0], correspondenceList[x*2][1],1,-correspondenceList[x*2][3]*correspondenceList[x*2][0],-correspondenceList[x*2][3]*correspondenceList[x*2][1],-correspondenceList[x*2][3]]

        V,D,K=np.linalg.svd(A)
        # reshape the min singular value into a 3 by 3 matrix
        h = np.reshape(K[8], (3, 3))

        # h[8] will be 1  it converts to homogenous homography
        h = (1 / h.item(8)) * h
        return h

def homography2(Corresponding_points):
    # print(Corresponding_points)
    correspondenceList=Corresponding_points
    h=np.zeros((3,3))
    A=np.zeros((8,9))

    for x in range (4):
        A[x*2]=[-correspondenceList[x][0][0], -correspondenceList[x][0][1],-1,0,0,0,correspondenceList[x][1][0]*correspondenceList[x][0][0],correspondenceList[x][1][0]*correspondenceList[x][0][1],correspondenceList[x][1][0]]
        A[x*2+1]=[0,0,0,-correspondenceList[x][0][0], -correspondenceList[x][0][1],-1,correspondenceList[x][1][1]*correspondenceList[x][0][0],correspondenceList[x][1][1]*correspondenceList[x][0][1],correspondenceList[x][1][1]]

    V,D,K=np.linalg.svd(A)
    # reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(K[8], (3, 3))
    # h[8] will be 1  it converts to homogenous homography
    h = (1 / h.item(8)) * h
    return h

# we need to use a method to define the threshold
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0][0], correspondence[0][1], 1]))  #obtain the x and y source
    estimatep2 = np.dot(h, p1) #calculate the estimated point by iner product of h and p1
    estimatep2 = (1/estimatep2.item(2))*estimatep2
    p2 = np.transpose(np.matrix([correspondence[1][0], correspondence[1][1], 1]))
    error = p2 - estimatep2
    return np.sum(np.abs(error)) #we can chhange it in the future

def ransac(corresponding_points, max_iterations=600):

    maxInliers = []
    finalH = None
    iteration = 0
    Current_count = 0
    while iteration < max_iterations:
        iteration += 1
        # cp_rand_index = np.random.choice(len(corresponding_points), int(np.floor(len(corresponding_points) / 2)))
        # 4 random matching index to calculate the homography
        firstRandom =random.choice(corresponding_points)
        secondRandom = random.choice(corresponding_points)
        thirdRandom = random.choice(corresponding_points)
        fourthRandom = random.choice(corresponding_points)
        ForRandomIndices = [firstRandom,secondRandom,thirdRandom,fourthRandom]
        # ForRandomIndices = [corresponding_points[4], corresponding_points[8], corresponding_points[12], corresponding_points[16]]
        # homography implement depend
        # rand_choosed_points=[]
        # for index in cp_rand_index:
        #     rand_choosed_points.append(corresponding_points[index])
        homography=homography2(ForRandomIndices)

        inliers = []
        Index_Inliers = []
        # comparing the values of the destination of match and estimated of the source
        for i in range(len(corresponding_points)):
            d = geometricDistance(corresponding_points[i], homography)
            if d < 1:
                inliers.append(corresponding_points[i])
                Index_Inliers.append(i)



        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            maxInliers_Index = Index_Inliers
            SelectedHomography = homography
            # finalH_cv = h_cv


        if len(maxInliers) > (len(corresponding_points)*.5):
            break
        # print(homography)


        '''
        # print(f"This is pc samples {pc_samples}")


        ########Find  Distance ###########
        plane_equation = model_fit_func(pc_samples)
        # print(f" Plane equation: {plane_equation}")
        Current_count = 0
        inner_loop_c = 0
        for index in range(len(point_cloud[0])):

            distance = (plane_equation[0] * point_cloud[0][index] + plane_equation[1] * point_cloud[1][index] +
                        plane_equation[2] * point_cloud[2][index] + plane_equation[3]) / np.sqrt(
                plane_equation[0] ** 2 + plane_equation[1] ** 2 + plane_equation[2] ** 2)
            # distance_eucledean=np.sqrt(distance[0]**2+distance[1]**2+distance[2]**2)
            if abs(distance) < tolerance:
                Current_count += 1
        if Current_count > best_ic:
            best_ic = Current_count
            best_model = plane_equation
    print(f"inlier count {best_ic}")
    print(f"plane_equation {best_model}")
    ##################
    '''
    return maxInliers_Index,maxInliers,SelectedHomography





def SIFT_MATCHES(img1,img2):
    sift = cv2.SIFT_create()
    kp1, img1_descriptor = sift.detectAndCompute(img1, None) # keypoints and descriptors
    kp2, img2_descriptor = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, True)
    # matches = bf.knnMatch(img1_descriptor,img2_descriptor, k=3)
    matches = bf.match(img1_descriptor, img2_descriptor)  # This is going to be replaced by kd-tree

    #  KD tree implementation
    tree = KDTree(img2_descriptor, leaf_size=2)
    dist, ind = tree.query(img1_descriptor, k=2)
    NNDRs = np.zeros(len(img1_descriptor))
    mins = np.zeros(len(img1_descriptor))
    for i in range(len(img1_descriptor)):
        NNDRs[i] = np.linalg.norm(img1_descriptor[i] - img2_descriptor[ind[i][0]]) / np.linalg.norm(img1_descriptor[i] - img2_descriptor[ind[i][1]])




    matchesKnn = []
    for i, nndr in enumerate(NNDRs):
        if nndr < .5:
            matchesKnn.append(cv2.DMatch(i,ind[i][0],0,np.linalg.norm(img1_descriptor[i] - img2_descriptor[ind[i][0]])))





    PairedPointsList = []
    for match in matchesKnn:
        p_source = kp1[match.queryIdx].pt
        p_dest = kp2[match.trainIdx].pt
        PairedPointsList.append([p_source, p_dest])

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matchesKnn, None)
    cv2.imwrite("matches_before_ransac.jpg", img3)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.show()

    return  PairedPointsList, matchesKnn, kp1, kp2

def Stitching(img1,img2,homography):
    img2_w , img2_h  = img2.shape[:2]
    img1_w , img1_h  = img1.shape[:2]
    #Corners_Img2 = [(0,0,),(img2_w,0),(0,img2_h),(img2_w,img2_h)]
    Img2_c0 = np.transpose(np.matrix([0, 0, 1]))
    Img2_c1 = np.transpose(np.matrix([img2_w,0, 1]))
    Img2_c2 = np.transpose(np.matrix([0,img2_h,1]))
    Img2_c3 = np.transpose(np.matrix([img2_w,img2_h,1]))
    Corners_Img2 = [Img2_c0,Img2_c1,Img2_c2,Img2_c3]
    Transformed_Corners_Img2 = []
    for corner in Corners_Img2:
        a = np.dot(homography, corner)
        a = (1/a.item(2))*a
        Transformed_Corners_Img2.append((a))
    t = cornersOfImg2

    # Canvas Cordinates
    c_x_min = np.min()



print("our code start to run...")
img1 = cv2.imread("im3_0.jpg")
img2 = cv2.imread("im3_1.jpg")
PairedPointsList, matches,kp1,kp2 = SIFT_MATCHES(img1,img2)


draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   flags=cv2.DrawMatchesFlags_DEFAULT)
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
# ransac(PairedPointsList)
Index_Inliers,Inliers,h = ransac(PairedPointsList)
canvas_img,projcanvas=Canvas(img1,img2,np.linalg.inv(h))
#Stitching(img1,img2,h)
# Good_Matches = matches.index(Index_Inliers)
Good_Matches = [matches[i] for i in Index_Inliers]   #found the indices of the best matches from ransac and obtain it from matches list
img4 = cv2.drawMatches(img1, kp1, img2, kp2, Good_Matches, None, **draw_params)
cv2.imwrite("matchaes_after_ransac.jpg",img4)
plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite("end1.jpg",canvas_img)
cv2.imwrite("end2.jpg",projcanvas)
print("program finished")
cv2.waitKey(0)
cv2.destroyAllWindows()





