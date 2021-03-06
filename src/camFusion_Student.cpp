
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    map<pair<int, int>, int> pointsInBoundingBoxes;
    map<pair<int, int>, int>::iterator pointsInBoundingBoxesIterator;

    const int columns = (int)(prevFrame.boundingBoxes.size());
    const int rows = (int)(currFrame.boundingBoxes.size());

    int previousBoundingBoxmatchingBoxId;
    int currentBoundingBoxmatchingBoxId;

    vector<std::vector<int>> TemporalMatchesList(columns, std::vector<int>(rows, 0));

    for (const auto& dMatch : matches) 
    {
        const int current_idx = dMatch.trainIdx;
        const int previous_idx = dMatch.queryIdx;

        // get points for previous and current frame
        const cv::Point2f currentPoint = currFrame.keypoints[current_idx].pt;
        const cv::Point2f previousPoint = prevFrame.keypoints[previous_idx].pt;

        int previousBoundingBoxmatchCounter = 0, currentBoundingBoxmatchCounter = 0;

        bool previousPointFound = false;
        bool currentPointFound = false;

        for (const BoundingBox& previousBoundingBox : prevFrame.boundingBoxes)
        {
            for (const BoundingBox& currentBoundingBox : currFrame.boundingBoxes)
            {
                if (previousBoundingBox.roi.contains(previousPoint))
                {
                    if (currentBoundingBox.roi.contains(currentPoint))
                    {
                        previousPointFound = true;
                        currentPointFound = true;
                    }
                }

                if( previousPointFound == true && currentPointFound == true)
                {
                    previousBoundingBoxmatchingBoxId = previousBoundingBox.boxID;
                    currentBoundingBoxmatchingBoxId = currentBoundingBox.boxID;
                    TemporalMatchesList.at(previousBoundingBoxmatchingBoxId).at(currentBoundingBoxmatchingBoxId)++;                    
                }
                previousPointFound = false;
                currentPointFound = false;
            }
        }
    }

    for (int columnIndex = 0; columnIndex < columns; columnIndex++) 
    {
        vector<int>::iterator maxElementIterator = max_element(begin(TemporalMatchesList.at(columnIndex)), 
                                                    end(TemporalMatchesList.at(columnIndex)));

        int rowIndex = std::distance(
            std::begin(TemporalMatchesList.at(columnIndex)), maxElementIterator);

        if (TemporalMatchesList.at(columnIndex).at(rowIndex) > 0)
        {
            bbBestMatches[prevFrame.boundingBoxes.at(columnIndex).boxID] = currFrame.boundingBoxes.at(rowIndex).boxID;
        }        
    }    
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1.0/frameRate; // time between two measurements in seconds
    // find closest distance to Lidar points 
    double minXPrev = 1e9;
    double minXCurr = 1e9;

    double averageMinXPrev = 0.0;
    double averageMinXCurr = 0.0;

    int counterMinXPrev = 0;
    int CounterMinXCurr = 0;

    const double laneWidth = 4.0;  // assumed width of the ego lane

    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it) 
    {
        if (std::abs(it->y) > laneWidth) { continue; }
        if(minXPrev>it->x)
        {
            averageMinXPrev = averageMinXPrev + it->x;
            counterMinXPrev++;
        }
    }

    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it) 
    {
        if (std::abs(it->y) > laneWidth) { continue; }
        if(minXCurr>it->x)
        {
            averageMinXCurr = averageMinXCurr + it->x;
            CounterMinXCurr++;
        }
    }

    minXPrev = averageMinXPrev/counterMinXPrev;
    minXCurr = averageMinXCurr/CounterMinXCurr;

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev-minXCurr);
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    bool previousPointFound = false;
    bool currentPointFound = false;
    
    double distanceMean = 0.0;
    int currentFrameIndex;
    int previousFrameIndex;

    std::vector<double> distance;
    
    boundingBox.keypoints.clear();
    boundingBox.kptMatches.clear();

    // First stage - just check to see if it's within ROI
    for (auto& dMatch : kptMatches)
    {   
        currentFrameIndex = dMatch.trainIdx; // trainIdx --> current frame
        previousFrameIndex = dMatch.queryIdx; // queryIdx --> previous frame
        cv::Point2f currentPoint = kptsCurr[currentFrameIndex].pt;
        cv::Point2f previousPoint = kptsPrev[previousFrameIndex].pt;

        if (boundingBox.roi.contains(previousPoint))
        {
            if (boundingBox.roi.contains(currentPoint))
            {
                previousPointFound = true;
                currentPointFound = true;
            }
        }

        if(currentPointFound == true)
        {
            distance.push_back(cv::norm(currentPoint - previousPoint));
        }

        previousPointFound = false;
        currentPointFound = false;
    }

    for (size_t i = 0; i < distance.size(); i++)
    {
        distanceMean = distanceMean + distance[i];
    }
    distanceMean = distanceMean / distance.size();
    distanceMean = distanceMean * 1.3;
        
    currentPointFound = false;
    for (auto point : kptMatches) 
    {
		auto &kptCurr{ kptsCurr.at(point.trainIdx) };

        if (boundingBox.roi.contains(kptCurr.pt))
        {
            previousFrameIndex = point.queryIdx;
            auto &kptPrev{ kptsPrev.at(previousFrameIndex) };

            if (cv::norm(kptCurr.pt - kptPrev.pt) < distanceMean) 
            {
                boundingBox.keypoints.push_back(kptCurr);
                boundingBox.kptMatches.push_back(point);
            }

        }
	}
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    long medianIndex;
    float medianDistanceRatio;
    vector<double> distanceRatios;

    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) 
    {
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        // inner kpt.-loop
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) 
        {
            double minDist = 100.0;
            
            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            // avoid division by zero
            if (distPrev > numeric_limits<double>::epsilon() && distPrev >= minDist) 
            {
                double distRatio = distCurr / distPrev;
                distanceRatios.push_back(distRatio);
            }
        }  // eof inner loop over all matched kpts
    }    // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distanceRatios.size() == 0) 
    { 
        TTC = NAN;
        return;
    }

    sort(begin(distanceRatios), end(distanceRatios));
    medianIndex = (long)(floor(distanceRatios.size() / 2.0)) ;


    // After removing outliers, find the median ratio
    medianDistanceRatio = distanceRatios.at(medianIndex);
    if (distanceRatios.size() % 2 == 0)
    {
        medianDistanceRatio = distanceRatios.at(medianIndex - 1) + distanceRatios.at(medianIndex);
        medianDistanceRatio = medianDistanceRatio / 2.0;
    }

    // Now calculate TTC
    const double dT = 1 / frameRate;
    TTC = -dT / (1.0f - medianDistanceRatio);
}