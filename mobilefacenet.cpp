#include "mobilefacenet.h"
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

//加载模型
Recognize::Recognize(const std::string &model_path) {

	std::string param_files = model_path + "/insightface_mfnet_mxnet.param";
	std::string bin_files = model_path + "/insightface_mfnet_mxnet.bin";
//    std::string param_files = model_path + "/mobilefacenet.param";
//    std::string bin_files = model_path + "/mobilefacenet.bin";
    std::cout<< "Model : " << param_files << std::endl;
	Recognet.load_param(param_files.c_str());
	Recognet.load_model(bin_files.c_str());
#if NCNN_VULKAN
    Recognet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN
}

Recognize::~Recognize() {

	Recognet.clear();
}

//执行Extractor，得到128维的特征
void Recognize::RecogNet(ncnn::Mat& img_) {

	ncnn::Extractor ex = Recognet.create_extractor();
	//设置单线程，多线程反而慢
	ex.set_num_threads(1);
	ex.set_light_mode(true);

    ex.input("data", img_);
	ncnn::Mat out;
    clock_t start_time = clock();
	ex.extract("fc1", out);
    clock_t finish_time = clock();

    //计算人脸识别时间
    double total_time = (double)(finish_time - start_time);
    std::cout << "Inference time : " << total_time / 1000<< "ms" << std::endl;

	feature_out.resize(128);
	for (int j = 0; j < 128; j++){
		feature_out[j] = out[j];
	}

    normalize(feature_out);
}

void Recognize::normalize(std::vector<float> &feature)
{
    float sum = 0.f;
    for(auto it = feature.begin(); it != feature.end(); ++it)
        sum += (*it) * (*it);

    sum = sqrt(sum);
    for(auto it = feature.begin(); it != feature.end(); ++it)
        *it /= sum;
}

//计算图片人脸特征值
void Recognize::start(const cv::Mat& img, std::vector<float>&feature) {

	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 112, 112);
	RecogNet(ncnn_img);
	feature = feature_out;
}

double calculSimilar(std::vector<float>& v1, std::vector<float>& v2){

	assert(v1.size() == v2.size());
//	//计算余弦距离，这里可以优化一下
//	double ret = 0.0, mod1 = 0.0, mod2 = 0.0;
//	for (std::vector<double>::size_type i = 0; i != v1.size(); ++i){
//		ret += v1[i] * v2[i];
//		mod1 += v1[i] * v1[i];
//		mod2 += v2[i] * v2[i];
//	}
//	return (ret / sqrt(mod1) / sqrt(mod2) ) ;

    //  计算欧式距离
    double dist = 0.0;
    for(int i=0; i<v1.size(); i++){
        dist += (v1[i]-v2[i])*(v1[i]-v2[i]);
    }
    return sqrt(dist);
}
