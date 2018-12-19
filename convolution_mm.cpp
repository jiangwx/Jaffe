#include "CNN.h"

static void fm2mm(float *fm, float *mm, layer l)
{
    for(int h=0;h<l.oh;h++)
    {
        for(int w=0;w<l.ow;w++)
        {
            for(int c=0;c<l.ic;c++)
            {
                for(int kh=0;kh<l.k;kh++)
                {
                    for(int kw=0;kw<l.k;kw++)
                    {

                        int fw = w*l.s - l.p + kw;//matrix coordinate to fm coordinate
                        int fh = h*l.s - l.p + kh;
                        int fm_index = c*l.ih*l.iw + fh*l.iw + fw;
                        int mm_index_t = (c*l.k*l.k + kh*l.k + kw)*l.ow*l.oh + h*l.ow + w;//transpose the matrix in this procedure

                        if((fw < 0)||(fh < 0)||(fw > (l.iw-1))||(fh > (l.ih-1)))
                            mm[mm_index_t]=0;
                        else
                            mm[mm_index_t]=fm[fm_index];
                    }
                }
            }
        }
    }
}

static void add_bias(float *ifm, float *ofm, float *bias, layer l)
{
    for (int oc = 0; oc < l.oc; oc++)
    {
        for (int i = 0; i < l.ow*l.oh; i++)
        {
            ofm[oc*l.ow*l.oh + i] = ifm[oc*l.ow*l.oh + i] + bias[oc];
        }
    }
}

static void gemm(float *im, float *weight, float* om, layer l)
{
    const float alpha=1;
    const float beta=0;
    int M = l.oc;
    int N = l.ow*l.oh;
    int K = l.ic*l.k*l.k;
    int lda=K;//A的列
    int ldb=N;//B的列
    int ldc=N;//C的列

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, weight, lda, im, ldb, beta, om, ldc);

}

void convolution_mm(float *ifm, float *ofm, float *weight, float *bias, float weight_scale, float data_scale, layer l)
{
    DTYPE tmp;
    int out;
    static float* ifmf=(float*)calloc(l.ih*l.iw*l.ic, sizeof(float));
    static float* weightf=(float*)calloc(l.k*l.k*l.ic*l.oc, sizeof(float));
    static float* ofmf=(float*)calloc(l.oh*l.ow*l.oc, sizeof(float));
    for(int i=0;i<l.ih*l.iw*l.ic;i++)   
    {
        tmp=(DTYPE)(ifm[i]*data_scale);
        printf("ifm[%d]=%d\n",i,tmp);
        ifmf[i]=(float)tmp;

    }
    for(int i=0;i<l.k*l.k*l.ic*l.oc;i++) 
    {
        tmp=(DTYPE)(weight[i]*weight_scale);
        weightf[i]=(float)tmp;
        printf("weight[%d]=%d\n",i,tmp);
    }

    if(l.k==1)
    {
        gemm(ifmf,weightf,ofmf,l);
        for(int i=0;i<l.oh*l.ow*l.oc;i++) 
        {
            out=(int)ofmf[i];
            ofmf[i]=(float)out/(weight_scale*data_scale); 
        }
    }
    else
    {
        //transpose input feature map to matrix
        float* immf=(float*)calloc(l.oh*l.ow*l.ic*l.k*l.k, sizeof(float));
        fm2mm(ifmf,immf,l);
        gemm(immf,weightf,ofmf,l);
        free(immf);
        for(int i=0;i<l.oh*l.ow*l.oc;i++) 
        {
            out=(int)ofmf[i];
            ofmf[i]=(float)out/(weight_scale*data_scale); 
        }
    }
    add_bias(ofmf,ofm,bias,l);
}