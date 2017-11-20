#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cstddef>
#include <cerrno>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <iostream>
#include <fstream>
using namespace std;

#define MAX_NEURON 6
#define MAX_DATA_SIZE 1000

FILE * similarity_file;
FILE * data_file;
FILE * output_file;
int dataSize = 100;
int maxEpoch = 5000;
const int layer_num = 5;
const int pattern_dimension = 2;

int* layer_neurons;
long double* patterns;
int** similarity_matrix;
long double *** weights;

long double * deltaVec[4];
long double  ** prev_output;
long double **  current_output;

int current_layer = 0;
pair<int,int> min_pair;
pair<int,int> max_pair;
long double  current_longest_distance;
long double current_shortest_distance;
long double  attractionRate = 0.0001;
long double  repelRate = 0.1;

random_device rand_gen;
random_device rand_gen2;
normal_distribution<long double > gaussian(0,1.0);
uniform_int_distribution<int> random_int(0,dataSize-1);
void CalcPairs(bool b);
long double  Calc_Norm_2( long double * Y1 , long double  * Y2 , int layer);
void ShowError();

int x;
int* label;

bool randomSelect = false;
ifstream fin;
ofstream fout;

int main(){

    unsigned seed;
    seed = (unsigned)time(NULL); // 取得時間序列
    srand(seed);

    int m;
    label = (int*) malloc(sizeof(int)*dataSize);
    layer_neurons = (int*) malloc(sizeof(int) * (layer_num+1 ));
    layer_neurons[0] = pattern_dimension;
    layer_neurons[1] = 5;
    layer_neurons[2] = 5;
    layer_neurons[3] = 5;
    layer_neurons[4] = 5;
    layer_neurons[5] = 5;

    patterns = (long double *) malloc( sizeof(long double ) * pattern_dimension * dataSize);
    similarity_matrix = (int**) malloc(sizeof(int*) * dataSize );
    for(int i=0;i<dataSize;i++)
    	similarity_matrix[i] = (int*) malloc(sizeof(int) * dataSize);


    weights = (long double ***) malloc(sizeof(long double **) * layer_num );
    for(int i=0 ;i<layer_num;i++){
    	weights[i] = (long double **) malloc(sizeof(long double *) * layer_neurons[i+1] );
    	for(int j=0;j< layer_neurons[i+1];j++)
    		weights[i][j] = (long double *) malloc(sizeof(long double ) * (layer_neurons[i]+1));
    }

    prev_output = (long double **) malloc(sizeof(long double *) * (dataSize+1));
    for(int i=0;i<dataSize;i++){
    	prev_output[i] = (long double *) malloc(sizeof(long double ) * MAX_NEURON);
    }

    current_output = (long double **) malloc(sizeof(long double *) * MAX_DATA_SIZE);
	for(int i=0;i<dataSize;i++)
		current_output[i] = (long double *) malloc(sizeof(long double ) * MAX_NEURON);


    //cout.precision(15);
    fin.open("hw2pt.dat");
    //data_file = fopen("hw2pt.dat","r");
    for(int i=0;i<dataSize;i++)
    	for(int j=0;j<pattern_dimension;j++){
             fin >> patterns[ i*pattern_dimension + j  ];
             cout<<patterns[ i*pattern_dimension + j  ]<<endl;
    	}
    fin.close();


    fout.open("output.txt");


    similarity_file = fopen("hw2class.dat","r");
    for(int i=0;i<dataSize;i++)
    	for(int j=0;j<dataSize;j++)
    		fscanf(similarity_file,"%d",&similarity_matrix[i][j]);

    for(int i=0;i<dataSize;i++)
        if(similarity_matrix[0][i]==1)
            label[i] = 1;
        else
            label[i] = 0;

    fclose(similarity_file);

    for(int i=0;i<dataSize;i++){
    	for(int j=0;j<pattern_dimension;j++){
    		prev_output[i][j] = patterns[ i*pattern_dimension + j];
    	}
    	prev_output[i][pattern_dimension] = -1.0;
    	//printf("pattern%d %f %f\n",i,prev_output[i][0],prev_output[i][1]);
    }


    for(int i=0;i<layer_num;i++){
        for(int j=0;j< layer_neurons[i+1] ;j++){
        for(int k=0;k< layer_neurons[i] + 1;k++){
            weights[i][j][k] =  (long double )rand()/(long double )(RAND_MAX/2) -1.0;
        }
        }
    }



    // printf("6\n");
    ShowError();

    for(current_layer =0; current_layer < layer_num; current_layer++){
        //randomSelect = true;
    	for(x=0;x<maxEpoch;x++){


                if(x==maxEpoch-1)
                    CalcPairs((true));
                else
                    CalcPairs(true);

                long double minnorm,maxnorm;

                minnorm = 0;
                maxnorm = 0;

                for(int i=0;i<layer_neurons[current_layer+1];i++){
                    minnorm += (current_output[min_pair.first][i] - current_output[min_pair.second][i]) * (current_output[min_pair.first][i] - current_output[min_pair.second][i]);
                    maxnorm += (current_output[max_pair.first][i] - current_output[max_pair.second][i]) * (current_output[max_pair.first][i] - current_output[max_pair.second][i]);

                }
                 minnorm = sqrt(minnorm);
                 maxnorm = sqrt(maxnorm);


    			for(int i=0;i<layer_neurons[current_layer+1] ;i++){
    				for(int j=0;j<layer_neurons[current_layer] + 1;j++){

    					int i1 = max_pair.first;
    					int i2 = max_pair.second;

    					long double  i1j = j== layer_neurons[current_layer] ? -1.0 : prev_output[i1][j];
    					long double  i2j = j== layer_neurons[current_layer] ? -1.0 : prev_output[i2][j];

    					long double  delta;
    					delta =
    					attractionRate *(
    					 (current_output[i1][i] - current_output[i2][i])/maxnorm * (1.0 - current_output[i1][i]* current_output[i1][i]) * i1j
    					   - (current_output[i1][i] - current_output[i2][i])/maxnorm * (1.0 - current_output[i2][i]* current_output[i2][i]) * i2j
    					) ;

    					i1 = min_pair.first;
    					i2 = min_pair.second;
    					i1j = j== layer_neurons[current_layer] ? -1.0 : prev_output[i1][j];
    					i2j = j== layer_neurons[current_layer] ? -1.0 : prev_output[i2][j];

    					delta += repelRate *(
    					  -(current_output[i1][i] - current_output[i2][i])/minnorm * (1.0 - current_output[i1][i]* current_output[i1][i]) * i1j
    					   + (current_output[i1][i] - current_output[i2][i])/minnorm * (1.0 - current_output[i2][i]* current_output[i2][i]) * i2j
    					);

    					weights[current_layer][i][j] -= delta;

    				}
    			}



    		if(current_longest_distance < 0.001 && current_shortest_distance > 4 * layer_neurons[current_layer+1])
    			break;

    	}

    	for(int i=0;i<dataSize;i++){
    		for(int j=0;j<layer_neurons[current_layer+1];j++)
    			prev_output[i][j] = current_output[i][j];
            prev_output[i][layer_neurons[current_layer+1]] = -1.0;
            fout << "pattern "<<i<<" ( "<<prev_output[i][0]<<" "<<prev_output[i][1]<<" "<<prev_output[i][2]<<" "<<prev_output[i][3]<<" "<<prev_output[i][4]<<")"<<endl;
            fout<< "( "<< (prev_output[i][0] >0.5? 1:0) << ","<< (prev_output[i][1] >0.5? 1:0)<<","<<(prev_output[i][2] >0.5? 1:0) <<","<<(prev_output[i][3] >0.5? 1:0)<<","<<(prev_output[i][4] >0.5? 1:0)<<")"<<" "<<label[i] <<endl;
    	}





    	printf("weight:\n");
    	for(int i=0;i<layer_neurons[current_layer+1];i++){
            for(int j=0;j<layer_neurons[current_layer]+1;j++)
                fout<<weights[current_layer][i][j]<<" ";
                //printf("%llf ",weights[current_layer][i][j]);
           fout<<endl;
    	}
    	//attractionRate *= 100.0;
    	repelRate *= 10.0;
    	attractionRate *=10.0;

    	ShowError();
    }

    fout.close();
    return 0;
}

long double  Sigmoid(long double  x){
    return 1.0 / ( 1.0 + exp(-x));
}

long double HardLimit(long double x){
    return x>0.5 ? 1.0 : -1.0;
}

void CalcPairs(bool b){

	//printf("epoch%d\n",x);
	 //if(x<500)
     //  randomSelect = true;
    // else
     //  randomSelect = false;

	long double temp[MAX_NEURON];
	for(int i=0;i<dataSize;i++){

		for(int j=0; j< layer_neurons[current_layer+1];j++){
				long double  d;
				d = 0;
               // printf("i%d j%d\n",i,j);
                //printf("prevoutput%d");
				for(int k=0 ; k< layer_neurons[current_layer] + 1;k++){
					d += prev_output[i][k] * weights[current_layer][j][k];
					//printf("%f %f %f\n",prev_output[i][k],weights[current_layer][j][k],prev_output[i][k]* weights[current_layer][j][k]);
				}
				//printf("\n");
				if(b)
				d = tanh(d);
				temp[j] = d;
		}

		for(int j=0;j<layer_neurons[current_layer+1];j++){
			current_output[i][j] = temp[j];
		}
       // printf("pattern%d %f %f %f %f %f\n",i,current_output[i][0],current_output[i][1],current_output[i][2],current_output[i][3],current_output[i][4]);

	}


	long double  cur_min ;
	long double  cur_max ;
	cur_min = 1e9;
    cur_max = -1e9;

	for(int i=0;i<dataSize-1;i++){
		for(int j = i+1;j<dataSize;j++){
			long double dis;// = Calc_Norm_2( current_output[i] , current_output[j] ,current_layer +1);
			long double  total;
            total = 0;
            for(int k =0; k<layer_neurons[current_layer+1] ;k++){
                total += pow(current_output[i][k] - current_output[j][k] , 2.0);
            }
            dis = sqrt(total);
			if(similarity_matrix[i][j] == 1 && dis > cur_max){
				max_pair = std::make_pair(i,j);
				current_longest_distance = dis;
				cur_max = dis;
			}
			else if( similarity_matrix[i][j]==0 && dis < cur_min){
				min_pair = std::make_pair(i,j);
				current_shortest_distance = dis;
				cur_min = dis;
			}

		}
	}

	 if(randomSelect){
        max_pair = std::make_pair( (int)(rand() % 100) ,(int)(rand() % 100)  );
        min_pair = std::make_pair( (int)(rand() % 100) , (int)(rand() % 100)  );
    }
	fout<<"layer:"<<current_layer<<" epoch:"<<x<<" longest intra:" << current_longest_distance <<" shortest inter:" << current_shortest_distance<<endl;
    fout<<"shortest inter-class: "<<min_pair.first<<" "<<min_pair.second<<"  longest intra-class:"<<max_pair.first<<" "<<max_pair.second<<endl;
	//printf("layer:%d  epoch:%d  longest distance intra-class:%llf  shortest distance inter-class:%llf\n",current_layer,x,current_longest_distance,current_shortest_distance);
     //printf("shotest inter-class: %d %d  longest intra-class:%d %d\n",min_pair.first,min_pair.second,max_pair.first,max_pair.second);
    // max_pair = std::make_pair( (int)rand()%100 , (int)rand()%100);
    //min_pair = std::make_pair( (int)rand()%100 , (int)rand()%100);
    //printf("%d\n",(int)rand()%100 );


}

/*
double * ComputePatternOutput( int index ){

	double * output = (double) malloc(sizeof(double)*MAX_NEURON);
	double temp[MAX_NEURON];
	int i = current_layer;
	for(int j=0; j< layer_neurons[i+1];j++){
			double d;
			d = 0;
			for(int k=0 ; k< layer_neurons[i] ;k++){
				d += prev_output[index][k] * weights[i][k][j];
			}
			output[j] = d;
	}

	for(int i=0;i<pattern_dimension;i++)
		temp[i] = pattern[i];

	for(int i=0;i<current_layer+1;i++){

		for(int j=0; j< layer_neurons[i+1];j++){
			double d;
			d = 0;
			for(int k=0 ; k< layer_neurons[i] ;k++){
				d += temp[k] * weights[i][k][j];
			}
			temp2[j] = d;
		}

		for(int j=0;j<MAX_NEURON;j++)
			temp[j] = temp2[j];
	}
	for(int i=0;i<MAX_NEURON;i++)
		output[i] = temp[i];

	return output;
}
*/
void ShowError(){

    int errorNum ;
    int slot[1000];
    bool b[5];

    errorNum = 0;
    for(int i=0;i<dataSize;i++)
        slot[i] = -1;

    for(int i=0;i<dataSize;i++){
        //printf("222\n");
        for(int j=0;j<5;j++){
            b[j] = false;
        }

        for(int j=0;j<layer_neurons[current_layer];j++){

            if(prev_output[i][j]>0.5)
                b[j] = true;
            else
                b[j] = false;
        }
        int index;
        index = 0;
        for(int j=0;j<5;j++){
            if(!b[j])
                continue;
            int qq;
            qq=1;
            for(int k=0;k<j;k++)
                qq *=2;
            index += qq ;
        }
        if(slot[index]<0){
            slot[index] = label[i];
            fout<<"Pattern:("<<b[0]<<","<<b[1]<<","<<b[2]<<","<<b[3]<<","<<b[4]<<")"<<" label :"<<label[i]<<endl;
        }
        else if( slot[index] != label[i]){
            errorNum++;
            //printf("QQ\n");
        }


    }

    fout<<"Error Pattern Num:"<<errorNum;
    printf("errorNum:%d\n",errorNum);


    return;

}


long double  Calc_Norm_2( long double * Y1 , long double  * Y2 , int layer){
	long double  total;
	total = 0;
	for(int i =0; i<layer_neurons[layer] ;i++){
		total += (Y1[i] - Y2[i]) * (Y1[i] - Y2[i]);
	}
	//printf("qq:%f\n",sqrt(total));
	return sqrt(total);
}
