import numpy as np
import cv2 as cv
import os
import itertools

def IncludeRepeatMatches(pt_from, pt_to):
    l1 = list(pt_from[0])
    l2 = list(pt_from[1])
    l3 = list(pt_from[2])
    r1 = list(pt_to[0])
    r2 = list(pt_to[1])
    r3 = list(pt_to[2])
    return l1 == l2 or l1 == l3 or l2 == l3 or r1 == r2 or r1 == r3 or r2 == r3

def Distance(pt1, pt2):
    return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5

IdealPositions = {}

markup = input("Enter markup file name: ")
writefile = open('results.txt', 'a')

sum_norm_matches = 0
count = 0
sum_matches = 0
comm_count = 0

foldernames = input("Enter data folder name: ").split(' ')
for folder in foldernames:
    sputnic_images = []
    uav_images = []
    for filename in os.listdir(folder+"sputnic/"):
        sputnic_images.append(filename)
    for filename in os.listdir(folder+"uav/"):
        uav_images.append(filename)
    for file in uav_images:
        name = ''
        for f in sputnic_images:
            if f[4:10] == file[:6]:
                name = f

        img1 = cv.imread(folder +"uav/" + file)
        img2 = cv.imread(folder + "sputnic/" + name)

        img_y, img_x, col = img1.shape

        #convert images from color to grayscale
        gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
        gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

        #satellite image's intensity normalization
        gray2 = cv.equalizeHist(gray2)

        #obtain keypoints sets by SIFT
        sift = cv.SIFT_create(nfeatures=1000, nOctaveLayers=1, contrastThreshold=0.01, sigma=3)
        kp1, des1 = sift.detectAndCompute(gray1,None)
        img1=cv.drawKeypoints(gray1,kp1,img1)
        sift = cv.SIFT_create(nfeatures=4000, nOctaveLayers=1, contrastThreshold=0.01, sigma=3)
        kp2, des2 = sift.detectAndCompute(gray2,None)
        img2=cv.drawKeypoints(gray2,kp2,img2)

        #match keypoints
        bf = cv.BFMatcher(normType = cv.NORM_L2, crossCheck =True)
        matches = bf.match(des1,des2)
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        #build transformation matrix, accounting RANSAC verification
        M, mask = cv.findHomography(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold= 8.0)


        #if amount of validated pairs is not enough - repeat two previous steps
        if len(mask[mask > 0]) < 20:
            sift = cv.SIFT_create(nfeatures=6000, nOctaveLayers=5, contrastThreshold=0.0, edgeThreshold = 0, sigma=3)
            kp1, des1 = sift.detectAndCompute(gray1,None)#[50:-50]
            img1=cv.drawKeypoints(gray1,kp1,img1)

            sift = cv.SIFT_create(nfeatures=8000, nOctaveLayers=5, contrastThreshold=0.0, edgeThreshold = 0, sigma=3)
            kp2, des2 = sift.detectAndCompute(gray2,None)
            img2=cv.drawKeypoints(gray2,kp2,img2)

            bf = cv.BFMatcher(normType = cv.NORM_L2, crossCheck =True)
            matches = bf.match(des1,des2)
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold= 10.0)
        

        matchesMask = mask.ravel().tolist()
        h,w,c = img1.shape

        indices = matchesMask
        my_matches = list(itertools.compress(matches, matchesMask))
        trios = list(itertools.combinations(my_matches, 3))
        warp_mat = None
        cos_max = 100
        min_riot_diff = 3000

        #search for "the best" transformation matrix enumerating threes of matches
        for triple in trios:
            pt_from = np.array([list(map(int, kp1[triple[0].queryIdx].pt)), list(map(int, kp1[triple[1].queryIdx].pt)), list(map(int, kp1[triple[2].queryIdx].pt))])
            pt_to = np.array([list(map(int, kp2[triple[0].trainIdx].pt)), list(map(int, kp2[triple[1].trainIdx].pt)), list(map(int, kp2[triple[2].trainIdx].pt))])
            if IncludeRepeatMatches(pt_from, pt_to):
                continue

            W = cv.getAffineTransform(pt_from.astype(np.float32), pt_to.astype(np.float32))
            pts = np.float32([ [0, 0, w-1], [0, h-1, h-1], [1, 1, 1]])
            dst = W @ pts
            dst = dst.T
            err1 = Distance(dst[1], dst[0])/Distance(dst[2], dst[1])
            v1 = [dst[0][0] - dst[1][0], dst[0][1] - dst[1][1]]
            v2 = [dst[2][0] - dst[1][0], dst[2][1] - dst[1][1]]
            if Distance(v1, [0, 0]) == 0 or Distance(v2, [0, 0]) == 0:
                continue
            cosin = abs(v1[0]*v2[0] + v1[1]*v2[1])/(Distance(v1, [0, 0])*Distance(v2, [0, 0]))
            err2 = h/w
            
            if cosin < cos_max:
                cos_max = cosin
                min_riot_diff = abs(err1 - err2)
                warp_mat = W
            elif cos_max - cosin < 0.1:
                if abs(err1 - err2)< min_riot_diff:
                    min_riot_diff = abs(err1 - err2)
                    warp_mat = W               
            else:
                continue
        
        
        test_img = cv.warpAffine(gray1, warp_mat, (img1.shape[1], img1.shape[0]))

        pts = np.float32([ [0, 0, w-1, w-1], [0, h-1, h-1, 0], [1, 1, 1, 1]])
        dst = warp_mat @ pts
        dst = dst.T
        transform_quality = [((dst[i][0] - IdealPositions[file[:6]][i*2])**2 + (dst[i][1] - IdealPositions[file[:6]][i*2 + 1])**2)**0.5 for i in range(4)]
        
        for elem in transform_quality:
            sum_matches = sum_matches + elem
            comm_count = comm_count + 1
            line = str(round(elem, 6)) + ' '
            writefile.write(line)
        writefile.write('\n')

        #if error is within the acceptable range, take error into account
        if all(d < 100 for d in transform_quality):
            sum_norm_matches  = sum_norm_matches + sum(transform_quality)
            count = count + 4
        print(file)
        print(transform_quality)


if count >0:
    print(sum_matches/comm_count, sum_norm_matches/count)
else:
    print(sum_matches/comm_count, 0)
print(count)
