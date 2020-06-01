#include "tracker.h"


using namespace JPDAFTracker;

void Tracker::drawTracks(cv::Mat &_img, int frame) const
{
  std::stringstream ss;
  for(const auto& track : tracks_)
  {
    if(track->getId() != -1)
    {
      ss.str("");
      ss << track->getId();
      std::flush(ss);
      const cv::Point& p = track->getLastPrediction();
      cv::circle(_img, p, 8, track->getColor(), -1);
      //cv::ellipse(img, p, cv::Size(25, 50), 0, 0, 360, track->getColor(), 3);

      //----------------------------------------
      // NOTE: Pico código acá
      //----------------------------------------
      std::string filename = "tracks.csv";

      std::ofstream csvFile;

      csvFile.open(filename, std::fstream::app);

      //Add the data
      csvFile << ss.str();
      csvFile << ",";
      csvFile << std::to_string(p.x);
      csvFile << ",";
      csvFile << std::to_string(p.y);
      csvFile << ",";
      csvFile << -1;
      csvFile << ",";
      csvFile << std::to_string(frame);
      csvFile << "\n";
      
      // Close the file
      csvFile.close();
      //---------------------------------------------------------------------
      

      cv::putText(_img, ss.str(), p, cv::FONT_HERSHEY_SIMPLEX,
		0.50, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
  }
}

Eigen::MatrixXf Tracker::joint_probability(const Matrices& _association_matrices,
						const Vec2f& selected_detections)
{
  uint hyp_num = _association_matrices.size();
  Eigen::VectorXf Pr(_association_matrices.size());
  uint validationIdx = _association_matrices.at(0).rows();
  uint tracksize = tracks_.size();
  float prior;
  
  //Compute the total volume
  float V = 0.;
  for(const auto& track : tracks_)
  {
    V += track->getEllipseVolume();
  }

  for(uint i = 0; i < hyp_num; ++i)
  {
    //I assume that all the measurments can be false alarms
    int false_alarms = validationIdx ;
    float N = 1.;
    //For each measurement j: I compute the measurement indicator ( tau(j, X) ) 
    // and the target detection indicator ( lambda(t, X) ) 
    for(uint j = 0; j < validationIdx; ++j)
    {
      //Compute the MEASURAMENT ASSOCIATION INDICATOR      
      const Eigen::MatrixXf& A_matrix = _association_matrices.at(i).block(j, 1, 1, tracksize);
      const int& mea_indicator = A_matrix.sum();     
      ///////////////////////////////////////////////
      
      if(mea_indicator == 1)
      {
	//Update the total number of wrong measurements in X
	--false_alarms;
	
	//Detect which track is associated to the measurement j 
	//and compute the probability
	for(uint notZero = 0; notZero < tracksize; ++notZero)
	{
	  if(A_matrix(0, notZero) == 1)
	  {
	    const Eigen::Vector2f& z_predict = tracks_.at(notZero)->getLastPredictionEigen();
	    const Eigen::Matrix2f& S = tracks_.at(notZero)->S();
	    const Eigen::Vector2f& diff = selected_detections.at(j) - z_predict;
	    cv::Mat S_cv;
	    cv::eigen2cv(S, S_cv);
	    //const float& b = diff.transpose() * S.inverse() * diff;
	    cv::Mat z_cv(cv::Size(2, 1), CV_32FC1);
	    cv::Mat det_cv(cv::Size(2, 1), CV_32FC1);
	    z_cv.at<float>(0) = z_predict(0);
	    z_cv.at<float>(1) = z_predict(1);
	    det_cv.at<float>(0) = selected_detections.at(j)(0);
	    det_cv.at<float>(1) = selected_detections.at(j)(1);
	    const float& b = cv::Mahalanobis(z_cv, det_cv, S_cv.inv());
	    N = N / sqrt((2*CV_PI*S).determinant())*exp(-b);
	  }
	}
      }
      
    }
    
    const float& likelyhood = N / float(std::pow(V, false_alarms));
       
    if(param_.pd == 1)
    {
      prior = 1.;
    }
    else
    {
      //Compute the TARGET ASSOCIATION INDICATOR
      prior = 1.;
      for(uint j = 0; j < tracksize; ++j)
      {
	const Eigen::MatrixXf& target_matrix = _association_matrices.at(i).col(j+1);
	const int& target_indicator = target_matrix.sum();
	prior = prior * std::pow(param_.pd, target_indicator) * std::pow((1 - param_.pd), (1 - target_indicator));
      }
    }
    
    //Compute the number of events in X for which the same target 
    //set has been detected
    int a = 1;
    for(int j = 1; j <= false_alarms; ++j)
    {
      a = a * j;
    }
    
    Pr(i) = a * likelyhood * prior;
  }
  
  const float& prSum = Pr.sum();
  
  if(prSum != 0.)
    Pr = Pr / prSum; //normalization
    
  //Compute Beta Coefficients
  Eigen::MatrixXf beta(validationIdx + 1, tracksize);
  beta = Eigen::MatrixXf::Zero(validationIdx + 1, tracksize);
  
   
  Eigen::VectorXf sumBeta(tracksize);
  sumBeta.setZero();
  
  
  for(uint i = 0; i < tracksize; ++i)
  {
    for(uint j = 0; j < validationIdx; ++j)
    {
      for(uint k = 0; k < hyp_num; ++k)
      {
	beta(j, i) = beta(j, i) + Pr(k) * _association_matrices.at(k)(j, i+1);
      }
      sumBeta(i) += beta(j, i);
    }
    sumBeta(i) = 1 - sumBeta(i);
  }
 
  
  beta.row(validationIdx) = sumBeta;

  return beta;
}


Tracker::Matrices Tracker::generate_hypothesis(const Vec2f& _selected_detections, 
						    const cv::Mat& _q)
{
  uint validationIdx = _q.rows;
  //All the measurements can be generated by the clutter track
  Eigen::MatrixXf A_Matrix(_q.rows, _q.cols); 
  A_Matrix = Eigen::MatrixXf::Zero(_q.rows, _q.cols);
  A_Matrix.col(0).setOnes();
  Matrices tmp_association_matrices(MAX_ASSOC, A_Matrix);
  
  uint hyp_num = 0;
  //Generating all the possible association matrices from the possible measurements
    
  if(validationIdx != 0)
  {
    for(uint i = 0; i < _q.rows; ++i)
    {
      for(uint j = 1; j < _q.cols; ++j)
      {
	if(_q.at<int>(i, j)) // == 1
	{
	  tmp_association_matrices.at(hyp_num)(i, 0) = 0;
	  tmp_association_matrices.at(hyp_num)(i, j) = 1;
	  ++hyp_num;
	  if ( j == _q.cols - 1 ) continue;
	  for(uint l = 0; l < _q.rows; ++l)
	  {
	    if(l != i)
	    {
	      for(uint m = j + 1; m < _q.cols; ++m) // CHECK Q.COLS - 1
	      {
		if(_q.at<int>(l, m))
		{
		  tmp_association_matrices.at(hyp_num)(i, 0) = 0;
		  tmp_association_matrices.at(hyp_num)(i, j) = 1;
		  tmp_association_matrices.at(hyp_num)(l, 0) = 0;
		  tmp_association_matrices.at(hyp_num)(l, m) = 1;
		  ++hyp_num;
		} //if(q.at<int>(l, m))
	      }// m
	    } // if l != i
	  } // l
	} // if q(i, j) == 1
      } // j
    } // i
  } 
  /////////////////////////////////////////////////////////////////////////////////
  Matrices association_matrices(hyp_num + 1);
  std::copy(tmp_association_matrices.begin(), tmp_association_matrices.begin() + hyp_num + 1, 
	    association_matrices.begin());
  return association_matrices;
}


Tracker::VecBool Tracker::analyze_tracks(const cv::Mat& _q, const std::vector<Detection>& _detections)
{
  const cv::Mat& m_q = _q(cv::Rect(1, 0, _q.cols - 1, _q.rows));
  cv::Mat col_sum(cv::Size(m_q.cols, 1), _q.type(), cv::Scalar(0));

  VecBool not_associate(m_q.cols, true); //ALL TRACKS ARE ASSOCIATED
  for(uint i = 0; i < m_q.rows; ++i)
  {
    col_sum += m_q.row(i);
  }
  cv::Mat nonZero;
  col_sum.convertTo(col_sum, CV_8UC1);
  
  
  cv::Mat zero = col_sum == 0;
  cv::Mat zeroValues;
  cv::findNonZero(zero, zeroValues);
  
  for(uint i = 0; i < zeroValues.total(); ++i)
  {
    not_associate.at(zeroValues.at<cv::Point>(i).x) = false;
  }   
  return not_associate;
}