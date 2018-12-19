#include "darknet19_mbn.h"
static layer DNet[27] = {
        { conv_,"image",     224,224,3,224,224,3,0,0,0 },
        { conv_,"conv1",     224,224,3,224,224,32,3,1,1 },
        { maxpool_,"pool1",  224,224,32,112,112,32,2,2,0 },
        { conv_,"conv2",     112,112,32,112,112,64,3,1,1 },
        { maxpool_,"pool2",  112,112,64,56,56,64,2,2,0 },
        { conv_,"conv3",     56,56,64,56,56,128,3,1,1 },
        { conv_,"conv4",     56,56,128,56,56,64,1,1,0 },
        { conv_,"conv5",     56,56,64,56,56,128,3,1,1 },
        { maxpool_,"pool5",  56,56,128,28,28,128,2,2,0 },
        { conv_,"conv6",     28,28,128,28,28,256,3,1,1 },
        { conv_,"conv7",     28,28,256,28,28,128,1,1,0 },
        { conv_,"conv8",     28,28,128,28,28,256,3,1,1 },
        { maxpool_,"pool8",  28,28,256,14,14,256,2,2,0 },
        { conv_,"conv9",     14,14,256,14,14,512,3,1,1 },
        { conv_,"conv10",    14,14,512,14,14,256,1,1,0 },
        { conv_,"conv11",    14,14,256,14,14,512,3,1,1 },
        { conv_,"conv12",    14,14,512,14,14,256,1,1,0 },
        { conv_,"conv13",    14,14,256,14,14,512,3,1,1 },
        { maxpool_,"pool13", 14,14,512,7,7,512,2,2,0 },
        { conv_,"conv14",    7,7,512,7,7,1024,3,1,1 },
        { conv_,"conv15",    7,7,1024,7,7,512,1,1,0 },
        { conv_,"conv16",    7,7,512,7,7,1024,3,1,1 },
        { conv_,"conv17",    7,7,1024,7,7,512,1,1,0 },
        { conv_,"conv18",    7,7,512,7,7,1024,3,1,1 },
        { conv_,"conv19",    7,7,1024,7,7,30,1,1,0 },
        { avgpool_,"pool19", 7,7,30,1,1,30,7,1,0 },
        { softmax_,"softmax",1,1,30,1,1,30,0,0,0 }
};

enum darknet19_mbn_idx {
    image, conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool5, conv6,
    conv7, conv8, pool8, conv9, conv10, conv11, conv12, conv13, pool13,
    conv14, conv15, conv16, conv17, conv18, conv19, pool19, softmax
};

static char Name[]="darknet19_mbn";

static float* image_blob;
static float* conv1_blob;
static float* conv1_weight;
static float* conv1_bias;
static float* pool1_blob;
static float* conv2_blob;
static float* conv2_weight;
static float* conv2_bias;
static float* pool2_blob;
static float* conv3_blob;
static float* conv3_weight;
static float* conv3_bias;
static float* conv4_blob;
static float* conv4_weight;
static float* conv4_bias;
static float* conv5_blob;
static float* conv5_weight;
static float* conv5_bias;
static float* pool5_blob;
static float* conv6_blob;
static float* conv6_weight;
static float* conv6_bias;
static float* conv7_blob;
static float* conv7_weight;
static float* conv7_bias;
static float* conv8_blob;
static float* conv8_weight;
static float* conv8_bias;
static float* pool8_blob;
static float* conv9_blob;
static float* conv9_weight;
static float* conv9_bias;
static float* conv10_blob;
static float* conv10_weight;
static float* conv10_bias;
static float* conv11_blob;
static float* conv11_weight;
static float* conv11_bias;
static float* conv12_blob;
static float* conv12_weight;
static float* conv12_bias;
static float* conv13_blob;
static float* conv13_weight;
static float* conv13_bias;
static float* pool13_blob;
static float* conv14_blob;
static float* conv14_weight;
static float* conv14_bias;
static float* conv15_blob;
static float* conv15_weight;
static float* conv15_bias;
static float* conv16_blob;
static float* conv16_weight;
static float* conv16_bias;
static float* conv17_blob;
static float* conv17_weight;
static float* conv17_bias;
static float* conv18_blob;
static float* conv18_weight;
static float* conv18_bias;
static float* conv19_blob;
static float* conv19_weight;
static float* conv19_bias;
static float* pool19_blob;

static void conv_relu(float* ifm, float* ofm, float* weight, float* bias, layer l)
{
    float* conv_blob = (float*)malloc(l.oc*l.oh*l.ow*sizeof(float));
    convolution_mm(ifm,conv_blob,weight,bias,l);
    leaky_relu(conv_blob, ofm, l);
    free(conv_blob);
}

static int max(float *in,layer l)
{
    int index = 0;
    float tmp=in[0];
    for (int i = 1; i < l.oc; i++)
    {
        if (in[i] > tmp)
        {
            index = i;
            tmp = in[i];
        }
    }
    return index;
}

static void load_DNet()
{
    load_weight(conv1_weight, DNet[conv1], Name);
    load_bias(conv1_bias,DNet[conv1], Name);

    load_weight(conv2_weight, DNet[conv2], Name);
    load_bias(conv2_bias,DNet[conv2], Name);

    load_weight(conv3_weight, DNet[conv3], Name);
    load_bias(conv3_bias,DNet[conv3], Name);

    load_weight(conv4_weight, DNet[conv4], Name);
    load_bias(conv4_bias,DNet[conv4], Name);

    load_weight(conv5_weight, DNet[conv5], Name);
    load_bias(conv5_bias,DNet[conv5], Name);

    load_weight(conv6_weight, DNet[conv6], Name);
    load_bias(conv6_bias,DNet[conv6], Name);

    load_weight(conv7_weight, DNet[conv7], Name);
    load_bias(conv7_bias,DNet[conv7], Name);

    load_weight(conv8_weight, DNet[conv8], Name);
    load_bias(conv8_bias,DNet[conv8], Name);

    load_weight(conv9_weight, DNet[conv9], Name);
    load_bias(conv9_bias,DNet[conv9], Name);

    load_weight(conv10_weight, DNet[conv10], Name);
    load_bias(conv10_bias,DNet[conv10], Name);

    load_weight(conv11_weight, DNet[conv11], Name);
    load_bias(conv11_bias,DNet[conv11], Name);

    load_weight(conv12_weight, DNet[conv12], Name);
    load_bias(conv12_bias,DNet[conv12], Name);

    load_weight(conv13_weight, DNet[conv13], Name);
    load_bias(conv13_bias,DNet[conv13], Name);

    load_weight(conv14_weight, DNet[conv14], Name);
    load_bias(conv14_bias,DNet[conv14], Name);

    load_weight(conv15_weight, DNet[conv15], Name);
    load_bias(conv15_bias,DNet[conv15], Name);

    load_weight(conv16_weight, DNet[conv16], Name);
    load_bias(conv16_bias,DNet[conv16], Name);

    load_weight(conv17_weight, DNet[conv17], Name);
    load_bias(conv17_bias,DNet[conv17], Name);

    load_weight(conv18_weight, DNet[conv18], Name);
    load_bias(conv18_bias,DNet[conv18], Name);

    load_weight(conv19_weight, DNet[conv19], Name);
    memset(conv19_bias,0,sizeof(float)*DNet[conv19].oc);
}


void darknet_mbn_init()
{
    image_blob = (float*)malloc(DNet[image].oc*DNet[image].oh*DNet[image].ow*sizeof(float));

    conv1_blob = (float*)malloc(DNet[conv1].oc*DNet[conv1].oh*DNet[conv1].ow*sizeof(float));
    conv1_weight = (float*)malloc(DNet[conv1].ic*DNet[conv1].oc*DNet[conv1].k*DNet[conv1].k*sizeof(float));
    conv1_bias = (float*)malloc(DNet[conv1].oc*sizeof(float));

    pool1_blob = (float*)malloc(DNet[pool1].oc*DNet[pool1].oh*DNet[pool1].ow*sizeof(float));

    conv2_blob = (float*)malloc(DNet[conv2].oc*DNet[conv2].oh*DNet[conv2].ow*sizeof(float));
    conv2_weight = (float*)malloc(DNet[conv2].ic*DNet[conv2].oc*DNet[conv2].k*DNet[conv2].k*sizeof(float));
    conv2_bias = (float*)malloc(DNet[conv2].oc*sizeof(float));

    pool2_blob = (float*)malloc(DNet[pool2].oc*DNet[pool2].oh*DNet[pool2].ow*sizeof(float));

    conv3_blob = (float*)malloc(DNet[conv3].oc*DNet[conv3].oh*DNet[conv3].ow*sizeof(float));
    conv3_weight = (float*)malloc(DNet[conv3].ic*DNet[conv3].oc*DNet[conv3].k*DNet[conv3].k*sizeof(float));
    conv3_bias = (float*)malloc(DNet[conv3].oc*sizeof(float));

    conv4_blob = (float*)malloc(DNet[conv4].oc*DNet[conv4].oh*DNet[conv4].ow*sizeof(float));
    conv4_weight = (float*)malloc(DNet[conv4].ic*DNet[conv4].oc*DNet[conv4].k*DNet[conv4].k*sizeof(float));
    conv4_bias = (float*)malloc(DNet[conv4].oc*sizeof(float));

    conv5_blob = (float*)malloc(DNet[conv5].oc*DNet[conv5].oh*DNet[conv5].ow*sizeof(float));
    conv5_weight = (float*)malloc(DNet[conv5].ic*DNet[conv5].oc*DNet[conv5].k*DNet[conv5].k*sizeof(float));
    conv5_bias = (float*)malloc(DNet[conv5].oc*sizeof(float));

    pool5_blob = (float*)malloc(DNet[pool5].oc*DNet[pool5].oh*DNet[pool5].ow*sizeof(float));

    conv6_blob = (float*)malloc(DNet[conv6].oc*DNet[conv6].oh*DNet[conv6].ow*sizeof(float));
    conv6_weight = (float*)malloc(DNet[conv6].ic*DNet[conv6].oc*DNet[conv6].k*DNet[conv6].k*sizeof(float));
    conv6_bias = (float*)malloc(DNet[conv6].oc*sizeof(float));

    conv7_blob = (float*)malloc(DNet[conv7].oc*DNet[conv7].oh*DNet[conv7].ow*sizeof(float));
    conv7_weight = (float*)malloc(DNet[conv7].ic*DNet[conv7].oc*DNet[conv7].k*DNet[conv7].k*sizeof(float));
    conv7_bias = (float*)malloc(DNet[conv7].oc*sizeof(float));

    conv8_blob = (float*)malloc(DNet[conv8].oc*DNet[conv8].oh*DNet[conv8].ow*sizeof(float));
    conv8_weight = (float*)malloc(DNet[conv8].ic*DNet[conv8].oc*DNet[conv8].k*DNet[conv8].k*sizeof(float));
    conv8_bias = (float*)malloc(DNet[conv8].oc*sizeof(float));

    pool8_blob = (float*)malloc(DNet[pool8].oc*DNet[pool8].oh*DNet[pool8].ow*sizeof(float));

    conv9_blob = (float*)malloc(DNet[conv9].oc*DNet[conv9].oh*DNet[conv9].ow*sizeof(float));
    conv9_weight = (float*)malloc(DNet[conv9].ic*DNet[conv9].oc*DNet[conv9].k*DNet[conv9].k*sizeof(float));
    conv9_bias = (float*)malloc(DNet[conv9].oc*sizeof(float));

    conv10_blob = (float*)malloc(DNet[conv10].oc*DNet[conv10].oh*DNet[conv10].ow*sizeof(float));
    conv10_weight = (float*)malloc(DNet[conv10].ic*DNet[conv10].oc*DNet[conv10].k*DNet[conv10].k*sizeof(float));
    conv10_bias = (float*)malloc(DNet[conv10].oc*sizeof(float));

    conv11_blob = (float*)malloc(DNet[conv11].oc*DNet[conv11].oh*DNet[conv11].ow*sizeof(float));
    conv11_weight = (float*)malloc(DNet[conv11].ic*DNet[conv11].oc*DNet[conv11].k*DNet[conv11].k*sizeof(float));
    conv11_bias = (float*)malloc(DNet[conv11].oc*sizeof(float));

    conv12_blob = (float*)malloc(DNet[conv12].oc*DNet[conv12].oh*DNet[conv12].ow*sizeof(float));
    conv12_weight = (float*)malloc(DNet[conv12].ic*DNet[conv12].oc*DNet[conv12].k*DNet[conv12].k*sizeof(float));
    conv12_bias = (float*)malloc(DNet[conv12].oc*sizeof(float));

    conv13_blob = (float*)malloc(DNet[conv13].oc*DNet[conv13].oh*DNet[conv13].ow*sizeof(float));
    conv13_weight = (float*)malloc(DNet[conv13].ic*DNet[conv13].oc*DNet[conv13].k*DNet[conv13].k*sizeof(float));
    conv13_bias = (float*)malloc(DNet[conv13].oc*sizeof(float));

    pool13_blob = (float*)malloc(DNet[pool13].oc*DNet[pool13].oh*DNet[pool13].ow*sizeof(float));

    conv14_blob = (float*)malloc(DNet[conv14].oc*DNet[conv14].oh*DNet[conv14].ow*sizeof(float));
    conv14_weight = (float*)malloc(DNet[conv14].ic*DNet[conv14].oc*DNet[conv14].k*DNet[conv14].k*sizeof(float));
    conv14_bias = (float*)malloc(DNet[conv14].oc*sizeof(float));

    conv15_blob = (float*)malloc(DNet[conv15].oc*DNet[conv15].oh*DNet[conv15].ow*sizeof(float));
    conv15_weight = (float*)malloc(DNet[conv15].ic*DNet[conv15].oc*DNet[conv15].k*DNet[conv15].k*sizeof(float));
    conv15_bias = (float*)malloc(DNet[conv15].oc*sizeof(float));

    conv16_blob = (float*)malloc(DNet[conv16].oc*DNet[conv16].oh*DNet[conv16].ow*sizeof(float));
    conv16_weight = (float*)malloc(DNet[conv16].ic*DNet[conv16].oc*DNet[conv16].k*DNet[conv16].k*sizeof(float));
    conv16_bias = (float*)malloc(DNet[conv16].oc*sizeof(float));

    conv17_blob = (float*)malloc(DNet[conv17].oc*DNet[conv17].oh*DNet[conv17].ow*sizeof(float));
    conv17_weight = (float*)malloc(DNet[conv17].ic*DNet[conv17].oc*DNet[conv17].k*DNet[conv17].k*sizeof(float));
    conv17_bias = (float*)malloc(DNet[conv17].oc*sizeof(float));

    conv18_blob = (float*)malloc(DNet[conv18].oc*DNet[conv18].oh*DNet[conv18].ow*sizeof(float));
    conv18_weight = (float*)malloc(DNet[conv18].ic*DNet[conv18].oc*DNet[conv18].k*DNet[conv18].k*sizeof(float));
    conv18_bias = (float*)malloc(DNet[conv18].oc*sizeof(float));

    conv19_blob = (float*)malloc(DNet[conv19].oc*DNet[conv19].oh*DNet[conv19].ow*sizeof(float));
    conv19_weight = (float*)malloc(DNet[conv19].ic*DNet[conv19].oc*DNet[conv19].k*DNet[conv19].k*sizeof(float));
    conv19_bias = (float*)malloc(DNet[conv19].oc*sizeof(float));

    pool19_blob = (float*)malloc(DNet[pool19].oc*DNet[pool19].oh*DNet[pool19].ow*sizeof(float));
    load_DNet();
}

void darknet_mbn_close()
{
    free(image_blob);

    free(conv1_blob);
    free(conv1_weight);
    free(conv1_bias);

    free(pool1_blob);
	
	free(conv2_blob);
    free(conv2_weight);
    free(conv2_bias);
	
    free(pool2_blob);

	free(conv3_blob);
    free(conv3_weight);
    free(conv3_bias);
	
	free(conv4_blob);
    free(conv4_weight);
    free(conv4_bias);
	
	free(conv5_blob);
    free(conv5_weight);
    free(conv5_bias);
	
    free(pool5_blob);

	free(conv6_blob);
    free(conv6_weight);
    free(conv6_bias);
	
	free(conv7_blob);
    free(conv7_weight);
    free(conv7_bias);
	
	free(conv8_blob);
    free(conv8_weight);
    free(conv8_bias);
	
    free(pool8_blob);

	free(conv9_blob);
    free(conv9_weight);
    free(conv9_bias);
	
	free(conv10_blob);
    free(conv10_weight);
    free(conv10_bias);
	
	free(conv11_blob);
    free(conv11_weight);
    free(conv11_bias);
	
	free(conv12_blob);
    free(conv12_weight);
    free(conv12_bias);
	
	free(conv13_blob);
    free(conv13_weight);
    free(conv13_bias);
	
    free(pool13_blob);

	free(conv14_blob);
    free(conv14_weight);
    free(conv14_bias);
	
	free(conv15_blob);
    free(conv15_weight);
    free(conv15_bias);
	
	free(conv16_blob);
    free(conv16_weight);
    free(conv16_bias);
	
	free(conv17_blob);
    free(conv17_weight);
    free(conv17_bias);
	
	free(conv18_blob);
    free(conv18_weight);
    free(conv18_bias);

    free(conv19_blob);
    free(conv19_weight);
    free(conv19_bias);

    free(pool19_blob);
}


int darknet19_mbn(float* input)
{
    timeval start,end;
    gettimeofday(&start, NULL);
    load_fm(input,DNet[image],Name);
    conv_relu(input, conv1_blob, conv1_weight, conv1_bias, DNet[conv1]);
    maxpool(conv1_blob,pool1_blob,DNet[pool1]);
    conv_relu(pool1_blob, conv2_blob, conv2_weight, conv2_bias, DNet[conv2]);
    maxpool(conv2_blob,pool2_blob,DNet[pool2]);
    conv_relu(pool2_blob, conv3_blob, conv3_weight, conv3_bias, DNet[conv3]);
    conv_relu(conv3_blob, conv4_blob, conv4_weight, conv4_bias, DNet[conv4]);
    conv_relu(conv4_blob, conv5_blob, conv5_weight, conv5_bias, DNet[conv5]);
    maxpool(conv5_blob,pool5_blob,DNet[pool5]);
    conv_relu(pool5_blob, conv6_blob, conv6_weight, conv6_bias, DNet[conv6]);
    conv_relu(conv6_blob, conv7_blob, conv7_weight, conv7_bias, DNet[conv7]);
    conv_relu(conv7_blob, conv8_blob, conv8_weight, conv8_bias, DNet[conv8]);
    maxpool(conv8_blob,pool8_blob,DNet[pool8]);
    conv_relu(pool8_blob, conv9_blob, conv9_weight, conv9_bias, DNet[conv9]);
    conv_relu(conv9_blob, conv10_blob, conv10_weight, conv10_bias, DNet[conv10]);
    conv_relu(conv10_blob, conv11_blob, conv11_weight, conv11_bias, DNet[conv11]);
    conv_relu(conv11_blob, conv12_blob, conv12_weight, conv12_bias, DNet[conv12]);
    conv_relu(conv12_blob, conv13_blob, conv13_weight, conv13_bias, DNet[conv13]);
    maxpool(conv13_blob,pool13_blob,DNet[pool13]);
    conv_relu(pool13_blob, conv14_blob, conv14_weight, conv14_bias, DNet[conv14]);
    conv_relu(conv14_blob, conv15_blob, conv15_weight, conv15_bias, DNet[conv15]);
    conv_relu(conv15_blob, conv16_blob, conv16_weight, conv16_bias, DNet[conv16]);
    conv_relu(conv16_blob, conv17_blob, conv17_weight, conv17_bias, DNet[conv17]);
    conv_relu(conv17_blob, conv18_blob, conv18_weight, conv18_bias, DNet[conv18]);
    convolution(conv18_blob, conv19_blob, conv19_weight, conv19_bias, DNet[conv19]);
    avgpool(conv19_blob,pool19_blob,DNet[pool19]);
    int label = max(pool19_blob, DNet[pool19]);
    gettimeofday(&end, NULL);
    long us = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    check_fm(pool19_blob,DNet[pool19],Name);
    printf("darknet19_mbn took %lu us\n", us);
    return label;
}
