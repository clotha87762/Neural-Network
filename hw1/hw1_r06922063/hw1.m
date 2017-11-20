%%  clean the environment

clc;
clear all;
dirName = 'hw1_result//';
mkdir(dirName);

%%  Set the MLP Parameters

hiddenLayerNum = 2;
hiddenLayerNodeNum = [ 8 , 6];
outputLayerNum = 1;

smoothingParameter = 0.05;   % for momentum
learningRate = 0.15; 
learningRateDecaySpeed = 0.0001; % if using adaptive learning rate
maxEpoch = 10000;
minLearningRate = 0.05;

UsingMomentum = 1;
UsingLearningDecay = 0;
UsingRprop = 1;
drawIteration = 200;

%%  initial data processing

input_name = 'hw1data.dat';
inputs = importdata(input_name,'\t',6);
datas = inputs.data(:,1:length(inputs.data(1,:))-1);

label = inputs.data(:,length(inputs.data(1,:)));
if (min(label) <0)
	label(label<0) = 0;
else
    label = label - min(label);
end
currentLabel = zeros(size(label));

inputLayerNum = length(datas(1,:));



layerNodeNum = [ inputLayerNum hiddenLayerNodeNum outputLayerNum];
totalLayerNum = length(hiddenLayerNodeNum) + 2;

layerNodeNum(1:end-1) = layerNodeNum(1:end-1)+1;
datas = [ones(length(datas(:,1)),1) datas];

Weight = cell(1,totalLayerNum);
deltaWeight = cell(1,totalLayerNum);
for i=1:length(Weight)-1
   Weight{i} = -1.0 + 2.0 * rand(layerNodeNum(i) , layerNodeNum(i+1));
   Weight{i}(:,1) = 0;
   deltaWeight{i} = zeros(layerNodeNum(i), layerNodeNum(i+1));
end


ActivatedNode = cell(1,totalLayerNum);

for i=1:totalLayerNum
    ActivatedNode{i} = zeros(1,layerNodeNum(i));
end

BPError = ActivatedNode;

%{
labelOutput = zeros(length(label), 2);
for i=1:length(label)
    if (label(i) == 1)
        labelOutput(i,:) = [1 0];
    else
        labelOutput(i,:) = [0 1];
    end
end
%}
labelOutput = label;

Weight{end} =  ones(layerNodeNum(end), 1);
tempDeltaWeight = deltaWeight ; % for momentum;
resDeltaWeight = deltaWeight;
tempResDeltaWeight = deltaWeight;


%%  Back Propagate

MSE = -1 * ones(1,maxEpoch);
earlyTerminatedEpochs = -1;
zeroRMSReached = 0;
deltas_min = 0.00001;
deltas_max = 50;
res_plus = 1.2;
res_neg = 0.5;

for i = 1 : maxEpoch 
    
    % Forward Pass
    for data = 1:length(datas(:,1))
        ActivatedNode{1} = datas(data,:);
        for j=2:totalLayerNum
           ActivatedNode{j} =  ActivatedNode{j-1} * Weight{j-1};
           ActivatedNode{j} = Sigmoid(ActivatedNode{j});
           if j == totalLayerNum
               % do nothing
           else
               ActivatedNode{j}(1) = 1;
           end
        end

        BPError{totalLayerNum} = labelOutput(data,:) - ActivatedNode{totalLayerNum} ;

        for j = totalLayerNum -1 : -1 : 1
            gradient = Sigmoid_Derivitive( ActivatedNode{ j+1 } );
            for k = 1:length(BPError{j})
               BPError{j}(k) = sum ( BPError{j+1} .* gradient .* Weight{j}(k,:) );
            end
        end

        for j= totalLayerNum : -1 : 2
            derivitive = Sigmoid_Derivitive(ActivatedNode{j});
            deltaWeight{j-1} = deltaWeight{j-1} + ActivatedNode{j-1}' * (BPError{j} .* derivitive); 
        end
        
    end
    
       if UsingRprop ==1 % Handle Resilient Gradient Descent
        if (mod(i,200)==0) %Reset Deltas
            for j = 1:totalLayerNum
                resDeltaWeight{j} = learningRate*resDeltaWeight{j};
            end
        end
        for j = 1:totalLayerNum-1
            mult = tempResDeltaWeight{j} .* deltaWeight{j};
            resDeltaWeight{j}(mult > 0) = resDeltaWeight{j}(mult > 0) * res_plus; % Sign didn't change
            resDeltaWeight{j}(mult < 0) = resDeltaWeight{j}(mult < 0) * res_neg; % Sign changed
            resDeltaWeight{j} = max(deltas_min,resDeltaWeight{j});
            resDeltaWeight{j} = min(deltas_max, resDeltaWeight{j});

            tempResDeltaWeight{j} = deltaWeight{j};

            deltaWeight{j} = sign(deltaWeight{j}) .* resDeltaWeight{j};
        end
    end

    if UsingMomentum == 1
        for j=1:totalLayerNum
            deltaWeight{j}  =  learningRate * deltaWeight{j} + smoothingParameter*tempDeltaWeight{j};
        end
        tempDeltaWeight = deltaWeight;
    else
          for j=1:totalLayerNum
            deltaWeight{j}  =  learningRate * deltaWeight{j} ;
          end
    end
    
    for j=1:totalLayerNum-1
       Weight{j} = Weight{j} + deltaWeight{j};
       %deltaWeight{j} = 0 * deltaWeight{j}; % reset delta weight to zero
    end
    
    for j=1:length(deltaWeight)
       deltaWeight{j} = 0 * deltaWeight{j};
    end
    
    if UsingLearningDecay ==1
       learningRate =  max ( minLearningRate , learningRate - learningRateDecaySpeed );
    end
    
    % Calculate Current Output
    for j=1:length(datas(:,1))
        
       % display(datas(j,:));
        %temp = EvaluateNetwork(datas(j,:), ActivatedNode, Weight , 0);        
        temp = datas(j,:);
        for k=1:(totalLayerNum -1)
           temp = temp * Weight{k};
           temp = Sigmoid(temp);
           if k ~= (totalLayerNum -1)
               temp(1) = 1;
           end
        end
        %display(temp);
        if temp > 0.5
            currentLabel(j) = 1;
        else
            currentLabel(j) = 0;
        end
    end

     MSE(i) = sum((currentLabel-label).^2)/(length(datas(:,1)));
     if (MSE(i) ==  0 )
        %display('qq');
        zeroRMSReached = 1;
         
     end
    
     
     if (zeroRMSReached || mod(i,drawIteration)==0)
        unique_label = unique(label);
        training_colors = {'g.', 'r.'};
        separation_colors = {'y.', 'b.'};
        subplot(1,2,1);
        cla;
        hold on;
        title(['Boundary for epich ' int2str(i) '.']);

        margin = 0.05; step = 0.05;
        xlim([min(datas(:,2))-margin max(datas(:,2))+margin]);
        ylim([min(datas(:,3))-margin max(datas(:,3))+margin]);
        for x = min(datas(:,2))-margin : step : max(datas(:,2))+margin
            for y = min(datas(:,3))-margin : step : max(datas(:,3))+margin
                
               % outputs = EvaluateNetwork([1 x y], ActivatedNode, Weight, 0);
                temp = [1 x y ];
                for k=1:(totalLayerNum -1)
                   temp = temp * Weight{k};
                   temp = Sigmoid(temp);
                   if k ~= (totalLayerNum -1)
                       temp(1) = 1;
                   end
                end
                %display(outputs);
                %bound = (1+unipolarBipolarSelector)/2;
                outputs = temp;
              if (outputs > 0.5 )
                       plot(x, y, separation_colors{1}, 'markersize', 16);
              else
                        plot(x, y, separation_colors{2}, 'markersize', 16);
              end
                
            end
        end

        for j = 1:length(unique_label)
            points = datas(label==unique_label(j), 2:end);
            plot(points(:,1), points(:,2), training_colors{j}, 'markersize', 10);
        end
        axis equal;

        % Draw Mean Square Error
        subplot(1,2,2);
        MSE(MSE==-1) = [];
        plot([MSE(1:i)]);
        ylim([-0.01 0.6]);
        title('Mean Square Error');
        xlabel('­¡¥N¦¸¼Æ');
        ylabel('MSE');
        grid on;

        saveas(gcf, sprintf('hw1_result//fig%i.png', i),'jpg');
        %pause(0.03);
         
        if zeroRMSReached==1
            earlyTerminatedEpochs = i;
            break; 
        end    
         
     end
      display([int2str(i) ' Epochs done out of ' int2str(maxEpoch) ' Epochs. MSE = ' num2str(MSE(i)) ' Learning Rate = ' ...
        num2str(learningRate) '.']);
    
    nbrOfEpochs_done = i;
    if (zeroRMSReached)
        saveas(gcf, sprintf('hw1_result//Final Result for %s.png', dataFileName),'jpg');
        break;
    end
    
end
display(['Mean Square Error = ' num2str(MSE(nbrOfEpochs_done)) '.']);

%%
% Building Hidden Tree
binary_code = cell(length(datas(:,1)) , totalLayerNum+1 );  
for i = 1: length(datas(:,1))
   
   output = datas(i,:);
   binary_code{i,1} = output;
   binary_code{i,totalLayerNum+1} = label(i,1);
   
   for j= 1 : totalLayerNum -1
       output = output * Weight{j};
       output(output>0) = 1;
       output(output<0) = 0;
       %output = Sigmoid(output);
       
       if j ~= (totalLayerNum -1)
          output(1) = 1;
       end
       
       if j~= totalLayerNum-1
           x = output(2:end);
       else
           x= output;
       end
      
       %x(x>0.5)  = 1;
       %x(x<=0.5) = 0;
       
       binary_code{i,j+1} = x ; 
   end
   
    
end
save('hw1_result//code.mat','binary_code');
save('hw1_result//weight.mat','Weight');