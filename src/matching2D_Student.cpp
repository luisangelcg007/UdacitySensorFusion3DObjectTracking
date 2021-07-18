
#include <numeric>
#include "matching2D.hpp"

using namespace std;

static CollectedData collectedData;

// Find best matches for keypoints in two camera images based on several matching methods
CollectedData matchDescriptors( std::vector<cv::KeyPoint> &kPtsSource, 
                                std::vector<cv::KeyPoint> &kPtsRef, 
                                cv::Mat &descSource, cv::Mat &descRef,
                                std::vector<cv::DMatch> &matches, 
                                std::string descriptorType, 
                                std::string matcherType,
                                std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    // configure matcher
    double t;

    if (matcherType.compare("MAT_BF") == 0)
    {
        std::cout << "ANGEL0" << std::endl;
        int normType{ ((descriptorType.compare("DES_BINARY") == 0) ? cv::NORM_HAMMING : cv::NORM_L2) };
        std::cout << "ANGEL-1" << std::endl;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        std::cout << "BF matching cross-check = " << crossCheck << std::endl;
        std::cout << "ANGEL1" << std::endl;

        if (descRef.type() != CV_8U) 
        { 
        std::cout << "ANGEL2" << std::endl;
            descRef.convertTo(descRef, CV_8U);
        std::cout << "ANGEL3" << std::endl; 
        }
        std::cout << "ANGEL-2" << std::endl;
        if (descSource.type() != CV_8U) 
        { 
        std::cout << "ANGEL4" << std::endl;
            descSource.convertTo(descSource, CV_8U);
        std::cout << "ANGEL5" << std::endl;
        }
        std::cout << "ANGEL-3" << std::endl;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        std::cout << "ANGEL6" << std::endl;
        if (descRef.type() != CV_32F) 
        { 
        std::cout << "ANGEL7" << std::endl;
            descRef.convertTo(descRef, CV_32F); 
        std::cout << "ANGEL8" << std::endl;
        }
        if (descSource.type() != CV_32F) 
        { 
        std::cout << "ANGEL9" << std::endl;
            descSource.convertTo(descSource, CV_32F);
        std::cout << "ANGEL10" << std::endl;
        }

        std::cout << "ANGEL11" << std::endl;
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::cout << "ANGEL12" << std::endl;
        std::cout << "FLANN matching" << std::endl;
    }

    std::cout << "ANGEL-4" << std::endl;
    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        std::cout << "ANGEL-5" << std::endl;
        t = static_cast<double>(cv::getTickCount());
        std::cout << "ANGEL-6" << std::endl;
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        std::cout << "ANGEL-7" << std::endl;
        t = ((static_cast<double>(cv::getTickCount())) - t) / cv::getTickFrequency();
        std::cout << "ANGEL-8" << std::endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        std::cout << "ANGEL-9" << std::endl;
        std::vector<std::vector<cv::DMatch>> kNearestNeighborMatches;
        t = static_cast<double>(cv::getTickCount());
        
        std::cout << "ANGEL-10" << std::endl;
        matcher->knnMatch(descSource, descRef, kNearestNeighborMatches, 2);

        std::cout << "ANGEL-11" << std::endl;
        int count = 0;
        for (auto index{ std::begin(kNearestNeighborMatches) }; index != std::end(kNearestNeighborMatches); index = index+1) 
        {
            count++;
            std::cout << "count = " << count << std::endl;
            if ((*index).at(0).distance < (0.8 * (*index).at(1).distance)) 
            {
            std::cout << "count 2= " << count << std::endl;
                matches.push_back((*index).at(0));
            std::cout << "count 3= " << count << std::endl;
            }
        }

        std::cout << "ANGEL-12" << std::endl;
        t = ((static_cast<double>(cv::getTickCount())) - t) / cv::getTickFrequency();

        std::cout << "ANGEL-13" << std::endl;
    }
    
    collectedData.numKeyPoints = (int)matches.size();
    collectedData.elapsedTime = ((1000 * t) / 1.0);
    return collectedData ;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
CollectedData descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } 
    else
    {
        if (descriptorType.compare("ORB") == 0) 
        {
            extractor = cv::ORB::create();
        } 
        else
        {
            if (descriptorType.compare("FREAK") == 0) 
            {
                extractor = cv::xfeatures2d::FREAK::create();
            } 
            else
            {
                if (descriptorType.compare("AKAZE") == 0) 
                {
                    extractor = cv::AKAZE::create();
                } 
                else
                {
                    if (descriptorType.compare("SIFT") == 0) 
                    {
                        extractor = cv::xfeatures2d::SIFT::create();
                    } 
                    else
                    {
                        if (descriptorType.compare("BRIEF") == 0) 
                        {
                            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
                        }
                        else
                        {
                            /* code */
                        }
                        
                    }
                }
            }
        }
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

    collectedData.numKeyPoints = (int)keypoints.size();
    collectedData.elapsedTime = ((1000 * t) / 1.0);
    return collectedData ;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
CollectedData detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    collectedData.numKeyPoints = (int)keypoints.size();
    collectedData.elapsedTime = ((1000 * t) / 1.0);
    return collectedData ;
}

CollectedData detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    bool foundOverlap = false;
    int blockSize = 2;
    int apertureSize = 3;
    int thresh = 200;
    int scaledApertureSize =apertureSize * 2;
    int response;
    double k = 0.04;
    double overlapThreshold = 0.0;
    double t = (double)cv::getTickCount();

    cv::Mat dst = cv::Mat::zeros( img.size(), CV_32FC1 );
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );

    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            response = (int) dst_norm.at<float>(i,j);
            if( response > thresh )
            {
                cv::KeyPoint point;
                point.pt = cv::Point2f(i, j);
                point.size = scaledApertureSize;
                point.response = response;
                point.class_id = 0;

                foundOverlap = false;

                for (auto iterator = std::begin(keypoints); iterator != std::end(keypoints); iterator++) 
                {
                    if (cv::KeyPoint::overlap(point, (*iterator)) > overlapThreshold) 
                    {
                        foundOverlap = true;

                        if (point.response > (*iterator).response) 
                        {
                            *iterator = point;
                            break;
                        }
                    }
                }

                if (!foundOverlap) { keypoints.push_back(point); }
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    if (bVis) 
    {
        const cv::Mat visImage{ img.clone() };
        constexpr char windowName[]{ "Harris Detector Results" };

        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);

        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    collectedData.numKeyPoints = (int)keypoints.size();
    collectedData.elapsedTime = ((1000 * t) / 1.0);
    return collectedData ;
}

CollectedData detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    double t;

    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0) 
    {
        // TYPE_9_16, TYPE_7_12, TYPE_5_8
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(30, true, type);
    }
    else 
    {
        if (detectorType.compare("BRISK") == 0) 
        {
            detector = cv::BRISK::create();
        } 
        else 
        {
            if (detectorType.compare("ORB") == 0) 
            {
                detector = cv::ORB::create();
            } 
            else
            {
                if (detectorType.compare("AKAZE") == 0) 
                {
                    detector = cv::AKAZE::create();

                } 
                else  
                {
                    if (detectorType.compare("SIFT") == 0)
                    {
                        detector = cv::xfeatures2d::SIFT::create();
                    }
                    else
                    {
                        {/* code */}
                    }
                }
            }
        }
    }

    t = static_cast<double>(cv::getTickCount());
    detector->detect(img, keypoints);
    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();

    if (bVis) 
    {
        const std::string windowName(detectorType + " detection results.");
        cv::Mat visImage{ img.clone() };
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 5);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    collectedData.numKeyPoints = (int)keypoints.size();
    collectedData.elapsedTime = ((1000 * t) / 1.0);
    return collectedData ;
}