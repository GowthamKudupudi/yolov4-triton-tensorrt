#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cmath>
#include <string>
#include <iostream>

#include "../utils/weights.h"

using namespace nvinfer1;
using namespace std;

#define USE_FP16
//#define USE_FP32

namespace yolov4tiny {

    // stuff we know about the network and the input/output blobs
    static const int INPUT_H = 1088;
    static const int INPUT_W = 1088;
    static const int CLASS_NUM = 12;

    static const int YOLO_FACTOR_1 = 32;
    static const std::vector<float> YOLO_ANCHORS_1 = { 10,14, 23,27, 37,58, 81,82, 135,169, 344,319};
    static const float YOLO_SCALE_XY_1 = 1.05f;
    static const int YOLO_NEWCOORDS_1 = 0;

    static const int YOLO_FACTOR_2 = 16;
    static const std::vector<float> YOLO_ANCHORS_2 = { 10,14, 23,27, 37,58, 81,82, 135,169, 344,319 };
    static const float YOLO_SCALE_XY_2 = 1.05f;
    static const int YOLO_NEWCOORDS_2 = 0;

    const char* INPUT_BLOB_NAME = "input";
    const char* OUTPUT_BLOB_NAME = "detections";

    IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
        float *gamma = (float*)weightMap[lname + ".weight"].values;
        float *beta = (float*)weightMap[lname + ".bias"].values;
        float *mean = (float*)weightMap[lname + ".running_mean"].values;
        float *var = (float*)weightMap[lname + ".running_var"].values;
        int len = weightMap[lname + ".running_var"].count;

        float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            scval[i] = gamma[i] / sqrt(var[i] + eps);
        }
        Weights scale{DataType::kFLOAT, scval, len};
        
        float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
        }
        Weights shift{DataType::kFLOAT, shval, len};

        float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            pval[i] = 1.0;
        }
        Weights power{DataType::kFLOAT, pval, len};

        weightMap[lname + ".scale"] = scale;
        weightMap[lname + ".shift"] = shift;
        weightMap[lname + ".power"] = power;
	std::cout << "addBatchNorm2d: " << "lname: " << lname << "; shift.count:" << shift.count << ";\n";
        IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
        assert(scale_1);
        return scale_1;
    }

    ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx) {
        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{s, s});
        conv1->setPaddingNd(DimsHW{p, p});

	std::cout << "convBnLeaky: addBatchNorm2d: linx: " << linx << std::endl;
        IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

        auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
        lr->setAlpha(0.1);

        return lr;
    }

    ILayer *upSample(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int channels)
    {
        float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * channels * 2 * 2));
        for (int i = 0; i < channels * 2 * 2; i++)
        {
            deval[i] = 1.0;
        }
        Weights deconvwts{DataType::kFLOAT, deval, channels * 2 * 2};
        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        IDeconvolutionLayer *deconv = network->addDeconvolutionNd(input, channels, DimsHW{2, 2}, deconvwts, emptywts);
        deconv->setStrideNd(DimsHW{2, 2});
        deconv->setNbGroups(channels);

        return deconv;
    }
    
    IPluginV2Layer * yoloLayer(INetworkDefinition *network, ITensor& input, int inputWidth, int inputHeight, int widthFactor, int heightFactor, int numClasses, const std::vector<float>& anchors, float scaleXY, int newCoords) {
        auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
        
        int yoloWidth = inputWidth / widthFactor;
        int yoloHeight = inputHeight / heightFactor;
        int numAnchors = anchors.size() / 2;

        PluginFieldCollection pluginData;
        std::vector<PluginField> pluginFields;
        pluginFields.emplace_back(PluginField("yoloWidth", &yoloWidth, PluginFieldType::kINT32, 1));
        pluginFields.emplace_back(PluginField("yoloHeight", &yoloHeight, PluginFieldType::kINT32, 1));
        pluginFields.emplace_back(PluginField("numAnchors", &numAnchors, PluginFieldType::kINT32, 1));
        pluginFields.emplace_back(PluginField("numClasses", &numClasses, PluginFieldType::kINT32, 1));
        pluginFields.emplace_back(PluginField("inputMultiplier", &widthFactor, PluginFieldType::kINT32, 1));
        pluginFields.emplace_back(PluginField("anchors", anchors.data(), PluginFieldType::kFLOAT32, anchors.size()));
        pluginFields.emplace_back(PluginField("scaleXY", &scaleXY, PluginFieldType::kFLOAT32, 1));
        pluginFields.emplace_back(PluginField("newCoords", &newCoords, PluginFieldType::kINT32, 1));
        pluginData.nbFields = pluginFields.size();
        pluginData.fields = pluginFields.data();

        IPluginV2 *plugin = creator->createPlugin("YoloLayer_TRT", &pluginData);
        ITensor* inputTensors[] = { &input };
        return network->addPluginV2(inputTensors, 1, *plugin);
    }

    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string &weightsPath) {
        INetworkDefinition* network = builder->createNetworkV2(0U);

        // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
        ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
        assert(data);

        std::map<std::string, Weights> weightMap = loadWeights(weightsPath);

        // define each layer.
        auto l0 = convBnLeaky(network, weightMap, *data, 32, 3, 2, 1, 0);
        auto l1 = convBnLeaky(network, weightMap, *l0->getOutput(0), 64, 3, 2, 1, 1);
        auto l2 = convBnLeaky(network, weightMap, *l1->getOutput(0), 64, 3, 1, 1, 2);
        ISliceLayer *l3 = network->addSlice(*l2->getOutput(0), Dims3{0, 0, 0}, Dims3{32, INPUT_W / 4, INPUT_H / 4}, Dims3{1, 1, 1});
        auto l4 = convBnLeaky(network, weightMap, *l3->getOutput(0), 32, 3, 1, 1, 4);
        auto l5 = convBnLeaky(network, weightMap, *l4->getOutput(0), 32, 3, 1, 1, 5);
	cout << "Adding route 6" << endl;
        ITensor *inputTensors6[] = {l5->getOutput(0), l4->getOutput(0)};
        auto cat6 = network->addConcatenation(inputTensors6, 2);
	cout << "Adding leaky convolution 7" << endl;
        auto l7 = convBnLeaky(network, weightMap, *cat6->getOutput(0), 64, 1, 1, 0, 7);
        cout << "Adding route 8" << endl;
        ITensor *inputTensors8[] = {l2->getOutput(0), l7->getOutput(0)};
        auto cat8 = network->addConcatenation(inputTensors8, 2);
        cout << "Adding maxpool 9" << endl;
        auto pool9 = network->addPoolingNd(*cat8->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
        pool9->setStrideNd(DimsHW{2, 2});
        cout << "Adding leaky convolution 10" << endl;
	auto l10 = convBnLeaky(network, weightMap, *pool9->getOutput(0), 128, 3, 1, 1, 10);
	cout << "Adding route 11\n";
        ISliceLayer *l11 = network->addSlice(*l10->getOutput(0), Dims3{0, 0, 0}, Dims3{64, INPUT_W / 8, INPUT_H / 8}, Dims3{1, 1, 1});
	cout << "Adding leaky convolution 12\n";
        auto l12 = convBnLeaky(network, weightMap, *l11->getOutput(0), 64, 3, 1, 1, 12);
	cout << "Adding leaky convolution 13\n";
        auto l13 = convBnLeaky(network, weightMap, *l12->getOutput(0), 64, 3, 1, 1, 13);
	cout << "Adding route 14\n";
        ITensor *inputTensors14[] = {l13->getOutput(0), l12->getOutput(0)};
        auto cat14 = network->addConcatenation(inputTensors14, 2);
	cout << "Adding leaky convolution 15\n";
        auto l15 = convBnLeaky(network, weightMap, *cat14->getOutput(0), 128, 1, 1, 0, 15);
	cout << "Adding route 16\n";
        ITensor *inputTensors16[] = {l10->getOutput(0), l15->getOutput(0)};
        auto cat16 = network->addConcatenation(inputTensors16, 2);
	cout << "Adding max pool 17\n";
        auto pool17 = network->addPoolingNd(*cat16->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
        pool17->setStrideNd(DimsHW{2, 2});
	cout << "Adding leaky convolution 18\n";
        auto l18 = convBnLeaky(network, weightMap, *pool17->getOutput(0), 256, 3, 1, 1, 18);
	cout << "Adding route 19\n";
        ISliceLayer *l19 = network->addSlice(*l18->getOutput(0), Dims3{0, 0, 0}, Dims3{128, INPUT_W / 16, INPUT_H / 16}, Dims3{1, 1, 1});
        cout << "Adding leaky convolution 20\n";
	auto l20 = convBnLeaky(network, weightMap, *l19->getOutput(0), 128, 3, 1, 1, 20);
	cout << "Adding leaky convolution 21\n";
        auto l21 = convBnLeaky(network, weightMap, *l20->getOutput(0), 128, 3, 1, 1, 21);
	cout << "Adding route 22\n";
        ITensor *inputTensors22[] = {l21->getOutput(0), l20->getOutput(0)};
        auto cat22 = network->addConcatenation(inputTensors22, 2);
        cout << "Adding leaky convolution 23\n";
	auto l23 = convBnLeaky(network, weightMap, *cat22->getOutput(0), 256, 1, 1, 0, 23);
	cout << "Adding route 24\n";
        ITensor *inputTensors24[] = {l18->getOutput(0), l23->getOutput(0)};
        auto cat24 = network->addConcatenation(inputTensors24, 2);
	cout << "Adding maxpool 25\n";
        auto pool25 = network->addPoolingNd(*cat24->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
        pool25->setStrideNd(DimsHW{2, 2});
	cout << "Adding leaky convolution 26\n";
        auto l26 = convBnLeaky(network, weightMap, *pool25->getOutput(0), 512, 3, 1, 1, 26);
	cout << "Adding leaky convolution 27\n";
        auto l27 = convBnLeaky(network, weightMap, *l26->getOutput(0), 256, 1, 1, 0, 27);
	cout << "Adding leaky convolution 28\n";
        auto l28 = convBnLeaky(network, weightMap, *l27->getOutput(0), 512, 3, 1, 1, 28);
	cout << "Adding linear convolution 29\n";
        IConvolutionLayer *conv29 = network->addConvolutionNd(*l28->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.29.conv.weight"], weightMap["model.29.conv.bias"]);
        assert(conv29);

        // 30 is a yolo layer
	cout << "Adding yolo layer 30\n";
        auto yolo30 = yoloLayer(network, *conv29->getOutput(0), INPUT_W, INPUT_H, YOLO_FACTOR_1, YOLO_FACTOR_1, CLASS_NUM, YOLO_ANCHORS_1, YOLO_SCALE_XY_1, YOLO_NEWCOORDS_1);
	cout << "Adding route 31\n";
        auto l31 = l27;
	cout << "Adding leaky convolution 32\n";
        auto l32 = convBnLeaky(network, weightMap, *l31->getOutput(0), 128, 1, 1, 0, 32);
        cout << "Adding upsample 33\n";
	auto deconv33 = upSample(network, weightMap, *l32->getOutput(0), 128);
        cout << "Adding route 34\n";
	ITensor *inputTensors34[] = {deconv33->getOutput(0), l23->getOutput(0)};
        auto cat34 = network->addConcatenation(inputTensors34, 2);
	cout << "Adding leaky convolution 35\n";
        auto l35 = convBnLeaky(network, weightMap, *cat34->getOutput(0), 256, 3, 1, 1, 35);
	cout << "Adding linear convolution 36\n";
        IConvolutionLayer *conv36 = network->addConvolutionNd(*l35->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.36.conv.weight"], weightMap["model.36.conv.bias"]);
        assert(conv36);

        // 37 is a yolo layer
	cout << "Adding yolo layer 37\n";
        auto yolo37 = yoloLayer(network, *conv36->getOutput(0), INPUT_W, INPUT_H, YOLO_FACTOR_2, YOLO_FACTOR_2, CLASS_NUM, YOLO_ANCHORS_2, YOLO_SCALE_XY_2, YOLO_NEWCOORDS_2);

	cout << "Adding route 38\n";
        ITensor* inputTensors38[] = {yolo30->getOutput(0), yolo37->getOutput(0)};
        auto cat38 = network->addConcatenation(inputTensors38, 2);
        cat38->getOutput(0)->setName(OUTPUT_BLOB_NAME);
        network->markOutput(*cat38->getOutput(0));

        // Build engine
        builder->setMaxBatchSize(maxBatchSize);
        config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    #ifdef USE_FP16
        config->setFlag(BuilderFlag::kFP16);
    #endif
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

        // Don't need the network any more
        network->destroy();

        // Release host memory
        for (auto& mem : weightMap)
        {
            free((void*) (mem.second.values));
        }

        return engine;
    }
}
