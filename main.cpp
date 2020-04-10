#include <iostream>
#include <time.h>
#include <fstream>
#include "net.h"
#include "mobilefacenet.h"


struct Face{
    std::basic_string<char> name;
    std::vector<float> feature;
};

struct SimilaristFace{
    std::basic_string<char> name;
    double score;
};

//生成人脸特征数据集
void get_feature2dataset(){

    char *model_path = "./models";
    Recognize recognize(model_path);

    std::string dataset_path = "./dataset.txt";
    std::ofstream fileout(dataset_path);

    std::string pattern_jpg;
    std::vector<cv::String> image_files;
    pattern_jpg = "./data/*.jpg";
    cv::glob(pattern_jpg, image_files);

    Face face;

    for(int i=0;i<image_files.size();i++){
        std::cout << image_files[i] << std::endl;
        cv::Mat img = cv::imread(image_files[i]);

        // face.name = (char* )image_files[i].data();
        int pos2 = image_files[i].find_last_of("_");
        int pos1 = image_files[i].find_last_of("/");
        std::string name(image_files[i].substr(pos1+1, pos2-pos1-1));
        face.name = name;
        std::cout << face.name << std::endl;

        fileout.open(dataset_path,std::ios::out|std::ios::app);
        if(!fileout.is_open()){
            std::cout << "特征数据集文件打开失败" << std::endl;
            exit(0);
        }
        fileout << face.name << " ";

        //计算人脸特征值
        recognize.start(img, face.feature);
        std::cout << "feature size: " << face.feature.size() << std::endl;

        float feature_data[128] = {0};
        for(int j=0;j<128;j++){
            feature_data[j] = face.feature[j];
            fileout << feature_data[j] << " ";
        }
        fileout << std::endl;

        fileout.close();
    }

}

//获取人脸特征数据集
void get_dataset2features(std::vector<Face> &dataset_faces){

    std::string dataset_path = "./dataset.txt";
    std::ifstream filein(dataset_path, std::ios::in);
    if(filein.fail()){
        std::cout << "特征数据集文件打开失败" << std::endl;
        exit(0);
    }

    while(!filein.eof()){
        Face face;
        filein >> face.name;
        //std::cout<<face.name<<std::endl;

        float feature_data[128] = {0};
        for(int j=0;j<128;j++){
            filein >> feature_data[j];
            face.feature.push_back(feature_data[j]);
        }

        //std::cout<<cv::Mat(face.feature,true)<<std::endl;
        dataset_faces.push_back(face);
    }

    filein.close();
}

//获取验证集人脸特征
void get_validation_data(std::vector<Face> &validation_faces, Recognize &recognize){

    //读取验证图片数据集
    std::string pattern_jpg;
    std::vector<cv::String> validation_images;
    pattern_jpg = "./validation/*.jpg";
    cv::glob(pattern_jpg, validation_images);

    //计算待验证
    for(int i=0; i<validation_images.size(); i++){
        Face validation_face;
        int pos2 = validation_images[i].find_last_of("_");
        int pos1 = validation_images[i].find_last_of("/");
        std::string name(validation_images[i].substr(pos1+1, pos2-pos1-1));
        validation_face.name = name;

        cv::Mat img = cv::imread(validation_images[i]);
        recognize.start(img, validation_face.feature);

        validation_faces.push_back(validation_face);
    }
}

//识别
void recognize_faces(std::vector<Face> &dataset_faces, std::vector<Face> &validation_faces, int &correct, int &total){

    double threshold = 0.3;

    for(int i=0; i<validation_faces.size(); i++){
        SimilaristFace similaristFace;
        similaristFace.name = "Nobody";
        similaristFace.score = 0;
        SimilaristFace face;

//        std::cout << validation_faces[i].feature.size() << " : ";
        for(int j=0; j<dataset_faces.size(); j++){
            face.name = dataset_faces[j].name;
//            std::cout << dataset_faces[j].feature.size() << " ";
            face.score = calculSimilar(validation_faces[i].feature, dataset_faces[j].feature);

            if(validation_faces[i].name != face.name){
                std::cout << validation_faces[i].name << " &&&&& " <<face.name << " : " << face.score << std::endl;
            }
//            if(validation_faces[i].name == face.name && face.score <= threshold){
//                std::cout << validation_faces[i].name << " &&&&& " <<face.name << " : " << face.score << std::endl;
//                correct += 1;
//            }
//
//            if(validation_faces[i].name != face.name && face.score > threshold){
//                std::cout << validation_faces[i].name << " &&&&& " <<face.name << " : " << face.score << std::endl;
//                correct += 1;
//            }

            total += 1;

//            std::cout << face.name << ":" << face.score << std::endl;

//            if(face.score > similaristFace.score){
//                similaristFace.name = face.name;
//                similaristFace.score = face.score;
//            }
        }
//        std::cout << validation_faces[i].name << " : " << similaristFace.name << " " << similaristFace.score << std::endl;
    }
}

double get_accuracy(Recognize &recognize){

    std::string img_dataset_path = "./lfw_112/";

    double threshold = 1.2;
    std::cout << "Threshold : " << threshold << std::endl;

    std::string pairs_path = "./pairs_path_1.txt";
    std::ifstream filein(pairs_path, std::ios::in);
    if(filein.fail()){
        std::cout << "pairs path 文件打开失败" << std::endl;
        exit(0);
    }

    int correct = 0;
    int pairs = 0;

    std::string img1_path;
    std::string img2_path;
    int flag = 0;
    std::vector<float> feature1;
    std::vector<float> feature2;
    double distance;

    while(!filein.eof()){
        pairs += 1;
        std::cout << "Compara : " << pairs << "......" << std::endl;

        filein >> img1_path;
        filein >> img2_path;
        filein >> flag;
        cv::Mat img1 = cv::imread(img_dataset_path + img1_path);
        cv::Mat img2 = cv::imread(img_dataset_path + img2_path);

        recognize.start(img1, feature1);
        recognize.start(img2, feature2);

        distance = calculSimilar(feature1, feature2);
        if(distance <= threshold && flag == 1){
            correct += 1;
        }

        if(distance > threshold && flag ==-1){
            correct += 1;
        }
    }
    filein.close();

    std::cout << "Total : " << pairs << std::endl;
    std::cout << "Correct : " << correct <<std::endl;

    return (double)correct/(double)pairs;
}


int main(){

    //生成人脸特征数据集
//    get_feature2dataset();

    //初始化模型
    char *model_path = "./models";
    Recognize recognize(model_path);

    //在lfw数据集上计算准确率
    double accuracy = get_accuracy(recognize);
    std::cout << "Accuracy : " << accuracy << std::endl;

//    //计算待验证
//    std::vector<Face> validation_faces;
//    get_validation_data(validation_faces, recognize);
////    for(int i=0;i<validation_faces.size();i++){
////        std::cout << "Name: " << validation_faces[i].name <<" ";
////        std::cout << validation_faces[i].feature.size() << std::endl;
////        std::cout << "Feature: " << cv::Mat(validation_faces[i].feature, true) << std::endl;
////    }
//
//    std::vector<Face> dataset_faces;
//    get_dataset2features(dataset_faces);
////    for(int i=0;i<dataset_faces.size();i++){
////        std::cout << "Name: " << dataset_faces[i].name << " ";
////        std::cout << dataset_faces[i].feature.size() << std::endl;
////        std::cout << "Feature: " << cv::Mat(dataset_faces[i].feature, true) << std::endl;
////    }
////
//    int correct = 0;
//    int total = 0;
//    recognize_faces(dataset_faces, validation_faces, correct, total);
//    std::cout << "correct: " << ((double)correct)/((double) total) << std::endl;
//    std::cout << "correct: " << correct << std::endl;

	return 0;
}