// 310_model.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace dnn;


struct Model_return
{
    Rect boxes;
    int object_id;
};

vector <string> getOutputsNames(const Net& net)
{
    //FILE* layersfile;

    //layersfile = fopen("out.txt", "w");

    //Net net_fix = net;

    static vector<String> names;
    if (names.empty())
    {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        cout << outLayers.size() << '\n';
        cout << layersNames[0] << '\n';
        //Ptr <dnn::Layer> strLayernames;
        //vector <Ptr <dnn::Layer>> StrLayers;
        for (size_t i = 0; i < outLayers.size(); ++i)
        {
            names[i] = layersNames[outLayers[i] - 1];
        }

        /*for (size_t i = 0; i < names.size(); i++)
        {
            cout << names[i] << '\n';
        }*/
        /*for (size_t i = 0; i < layersNames.size(); i++)
        {
            //cout << layersNames[i] << '\n';
            strLayernames = net_fix.getLayer(net_fix.getLayerId(layersNames[i]));
            StrLayers = net_fix.getLayerInputs(net_fix.getLayerId(layersNames[i]));
            //strLayernames->getDefaultName();
            //strLayernames->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));
            //cout << strLayernames << '\n';

            //fprintf(layersfile, "%s\n", layersNames[i]);
        }*/
        /*for (size_t i = 0; i < StrLayers.size(); i++)
        {
            cout << StrLayers[i] << '\n';
        }*/
        /*for (size_t i = 0; i < outLayers.size(); i++)
        {
            cout << outLayers[i] << '\n';
        }*/
    }

    //fclose(layersfile);

    return names;
}




vector<Model_return> model(char* video_path, char* model_path)
{
    Net net;
    Mat blob, frame;

    float confThreshold = 0.5; // Confidence threshold
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold

    string inputfile = "C:\\Users\\Бобур Ибрагимов\\source\\repos\\310_model\\videoplayback.mp4";
    //string outputfile = "C:\\Users\\Бобур Ибрагимов\\source\\repos\\310_model\\video.avi";
    //VideoWriter video;
    VideoCapture cap("videoplayback.mp4");

    net = readNetFromONNX("C:\\Users\\Бобур Ибрагимов\\source\\repos\\310_model\\number_plate.onnx");
    //net.setPreferableBackend(DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(DNN_TARGET_CPU);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

    static const string kwinName = "Object detection";
    namedWindow(kwinName, WINDOW_NORMAL);

    vector<Model_return> model_array;

    while (1)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Done processing !!!" << endl;
            //cout << "Output file is stored as " << outputfile << endl;
            //waitKey(300);
            break;
        }


        blob = blobFromImage(frame, 1.0 / 255, Size(640, 640), (0, 0, 0), true, false);
        net.setInput(blob);

        vector<Mat> outs;
        vector<String> names;
        
        //names = net.getUnconnectedOutLayersNames();
        //names = net.getLayerNames();
        names = getOutputsNames(net);
        if (names.size() != 0)
        {
            for (int i = 0; i < names.size(); i++)
            {
                cout << names[i] << '\n';
            }
        }
        net.forward(outs, names);
        
        imshow(kwinName, frame);

        //vector<Rect> boxes;
        
        Model_return model_return;

        for (size_t i = 0; i < outs.size(); ++i)
        {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    //classIds.push_back(classIdPoint.x);
                    //confidences.push_back((float)confidence);
                    //boxes.push_back(Rect(left, top, width, height));
                    model_return.boxes = Rect(left, top, width, height);
                    model_return.object_id = classIdPoint.x;
                    model_array.push_back(model_return);
                }
            }
        }
        //cout << boxes[boxes.size() - 1].width << '\n';
        //cout << boxes[boxes.size() - 1].height << '\n';

    }





    cap.release();
    //video.release();

    return model_array;

}


int main(char* video_path, char* model_path, vector<Model_return> struct_array)
{

    //vector<Model_return> model_array = model(video_path, model_path);
    struct_array = model(video_path, model_path);

    return 0;
}
