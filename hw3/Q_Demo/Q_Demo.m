function varargout = Q_Demo(varargin)

% MENU Application M-file for menu.fig
%    FIG = MENU launch menu GUI.
%    MENU('callback_name', ...) invoke the named callback.

% Last Modified by GUIDE v2.5 18-Mar-2017 04:30:49

if nargin == 0  % LAUNCH GUI

	fig = openfig(mfilename,'reuse');

	% Use system color scheme for figure:
	set(fig,'Color',get(0,'defaultUicontrolBackgroundColor'));

	% Generate a structure of handles to pass to callbacks, and store it. 
	handles = guihandles(fig);
	guidata(fig, handles);

    if nargout > 0
		varargout{1} = fig;
	end

elseif ischar(varargin{1}) % INVOKE NAMED SUBFUNCTION OR CALLBACK

	try
		[varargout{1:nargout}] = feval(varargin{:}); % FEVAL switchyard
	catch
		disp(lasterr);
	end

end


%| ABOUT CALLBACKS:
%| GUIDE automatically appends subfunction prototypes to this file, and 
%| sets objects' callback properties to call them through the FEVAL 
%| switchyard above. This comment describes that mechanism.
%|
%| Each callback subfunction declaration has the following form:
%| <SUBFUNCTION_NAME>(H, EVENTDATA, HANDLES, VARARGIN)
%|
%| The subfunction name is composed using the object's Tag and the 
%| callback type separated by '_', e.g. 'slider2_Callback',
%| 'figure1_CloseRequestFcn', 'axis1_ButtondownFcn'.
%|
%| H is the callback object's handle (obtained using GCBO).
%|
%| EVENTDATA is empty, but reserved for future use.
%|
%| HANDLES is a structure containing handles of components in GUI using
%| tags as fieldnames, e.g. handles.figure1, handles.slider2. This
%| structure is created at GUI startup using GUIHANDLES and stored in
%| the figure's application data using GUIDATA. A copy of the structure
%| is passed to each callback.  You can store additional information in
%| this structure at GUI startup, and you can change the structure
%| during callbacks.  Call guidata(h, handles) after changing your
%| copy to replace the stored original so that subsequent callbacks see
%| the updates. Type "help guihandles" and "help guidata" for more
%| information.
%|
%| VARARGIN contains any extra arguments you have passed to the
%| callback. Specify the extra arguments by editing the callback
%| property in the inspector. By default, GUIDE sets the property to:
%| <MFILENAME>('<SUBFUNCTION_NAME>', gcbo, [], guidata(gcbo))
%| Add any extra arguments after the last argument, before the final
%| closing parenthesis.

% --------------------------------------------------------------------
function varargout = alpha_sl_Callback(h, eventdata, handles, varargin)
set(handles.alpha,'String',...
    num2str(get(handles.alpha_sl,'Value')));

% --------------------------------------------------------------------
function varargout = beta_sl_Callback(h, eventdata, handles, varargin)
set(handles.beta,'String',...
    num2str(get(handles.beta_sl,'Value')));

% --------------------------------------------------------------------
function varargout = gamma_sl_Callback(h, eventdata, handles, varargin)
set(handles.gamma,'String',...
    num2str(get(handles.gamma_sl,'Value')));

% --------------------------------------------------------------------
function varargout = reinf_sl_Callback(h, eventdata, handles, varargin)
set(handles.reinf,'String',...
    num2str(get(handles.reinf_sl,'Value')));

% --------------------------------------------------------------------
% Quit button
function varargout = pushbutton2_Callback(h, eventdata, handles, varargin)
delete(handles.figure1);

% --------------------------------------------------------------------
function varargout = figure1_ResizeFcn(h, eventdata, handles, varargin)
% Stub for ResizeFcn of the figure handles.figure1.
disp('figure1 ResizeFcn not implemented yet.')

% --------------------------------------------------------------------
% It is the main start button for q-learning demo
function varargout = pushbutton1_Callback(h, eventdata, handles, varargin)
    opengl autoselect;
    global ALPHA BETA GAMMA LAMBDA
    ALPHA = get(handles.alpha_sl,'Value')  % learning rate parameter 
	BETA = get(handles.beta_sl,'Value')    % magnitude of noise added to choice 
	GAMMA = get(handles.alpha_sl,'Value')  % discount factor for future reinf 
    if (get(handles.radiobutton1,'Value')==0)
        BETA=0
    end  %if
    
    LAMBDA = 0.8;
	
    m=1.1;  %mass of cart + pole 
	mp=0.1; %mass of the pole
	g=9.8;  %重力加速度
	length=0.5;  %half length of pole
	Force=10;   %force =10N
	T=0.02;  % Update time interval
	% Define global variable
	NUM_BOX=162;    % Number of states sets to 162
    preaction = 0;
   	[pre_state,cur_state,pre_action,cur_action,x,v_x,theta,v_theta] = reset_cart(BETA);  % reset the cart pole to initial state
	q_val=zeros(162,2);
    
    V = zeros(162,1);
    
	h1=figure;    % this figure is cart-pole
	axis([-3 3 0 1.5]);
	set(gca, 'DataAspectRatio',[1 1 1]);
    set(h1, 'Position',[10 100 500 200]);
	set(h1,'DoubleBuffer','on')
	set(gca, 'PlotBoxAspectRatio',[1 1 1]);
    h2 = figure;   % this figure is Q-value
    set(h2,'Position',[510 50 400 400]);
    set(h2,'Renderer','OpenGL')
    view(40,-30);
    success=0;   % succesee 0 times
	reinf=0;
    trial=0;
    % prepare grid for
	[GX,GY]=meshgrid(1:18,1:9);
    GZ=zeros(9,18);
    best=0;
    trialArray = [];
    successArray = [];
    x_hat = zeros(162,1);
    
	while success<50000
        
        if trial > 10000
           break; 
        end
        %disp(success);
        [q_val,pre_state,pre_action,cur_state,cur_action,V] = get_action(x,v_x,theta,v_theta,reinf,q_val,pre_state,cur_state,pre_action,cur_action,ALPHA,BETA,GAMMA,x_hat,V);
        if (cur_action==1)   % push left
            F=-1*Force;
        else
            F=Force;    % push right
        end %if
        % Update the cart-pole state
        a_theta=(m*g*sin(theta)-cos(theta)*(F+mp*length*v_theta^2*sin(theta)))/(4/3*m*length-mp*length*cos(theta)^2);
        a_x=(F+mp*length*(v_theta^2*sin(theta)-a_theta*cos(theta)))/m;
        v_theta=v_theta+a_theta*T;
        theta=theta+v_theta*T;
        v_x=v_x+a_x*T;
        x=x+v_x*T;
        
        x_hat = x_hat.* LAMBDA;
        x_hat(cur_state) = x_hat(cur_state) * (1.0 - LAMBDA);
        
        % draw new state
        %{
        figure(h1);
        X=[x   x+cos(pi/2-theta)];
        Y=[0.2  0.2+sin(pi/2-theta)];
        obj=rectangle('Position',[x-0.4,0.1,0.8,0.1],...
              'facecolor','b');
        obj2=line(X,Y);  
        hold on
        if (F>0)
           obj3=plot(x-0.4,0.11,'r>');
        else
           obj3=plot(x+0.4,0.11,'r<');
        end %if
        pause(0.00000000001)
        delete(obj)
        delete(obj2)
        delete(obj3)
        %}
        % get new box
        [box] = get_box(x,v_x,theta,v_theta);
        if (box== -1)  % if fail
            reinf=get(handles.reinf_sl,'Value');
            predicted_value=0;
            %q_val(pre_state,pre_action)= q_val(pre_state,pre_action)+ ALPHA*(reinf+ GAMMA*predicted_value - q_val(pre_state,pre_action));
        	[q_val,V] = failed_update(q_val,V,reinf,predicted_value,pre_state,pre_action,cur_state,x_hat);
            [pre_state,cur_state,pre_action,cur_action,x,v_x,theta,v_theta] = reset_cart(BETA);  % reset the cart pole to initial state
            trial=trial+1;
            if (success>best)
                best=success;
                disp(strcat('best: ',int2str(best),' trial:',int2str(trial)));
            end
            trialArray = [ trialArray trial];
            successArray = [successArray success];
            success=0;
            %{
            figure(h1);
            title(strcat('Trials  ',num2str(trial),', Best success : ',num2str(best)));
            % draw q-val figure
            figure(h2);
            title('Q-value mesh : up is push right, bottom is push left');
            for i=1:2
                st=1;
                for j=1:9
                    for k=1:18
                        GZ(j,k)=q_val(st,i)+i;  
                        st=st+1;
                    end % k
                end  % j
                surf(GX,GY,GZ);
                view(30,-15);
                hold on;
            end  %for i
            %}
        else
            success=success+1;
            reinf=0;
        end  %if (box
        
	end %while
            trialArray = [trialArray trial];
            successArray = [successArray success];
            figure();
            plot(trialArray,successArray);
            xlabel('trial');
            ylabel('iteration achieved');
            saveas(gcf,'iteration.jpg');
            
            
            best = success;
            disp(strcat('best: ',int2str(best),' trial:',int2str(trial)));
            figure(h1);
            X=[x   x+cos(pi/2-theta)];
            Y=[0.2  0.2+sin(pi/2-theta)];
            obj=rectangle('Position',[x-0.4,0.1,0.8,0.1],...
                  'facecolor','b');
            obj2=line(X,Y);  
            hold on
            if (F>0)
               obj3=plot(x-0.4,0.11,'r>');
            else
               obj3=plot(x+0.4,0.11,'r<');
            end %if
            pause(0.00000000001)
            %delete(obj)
            %delete(obj2)
            %delete(obj3)
        
        
            figure(h1);
            title(strcat('Trials  ',num2str(trial),', Best success : ',num2str(best)));
            saveas(gcf,'Simulated cart.jpg');
            
            % draw q-val figure
            figure(h2);
            title('Q-value mesh : up is push right, bottom is push left');
            for i=1:2
                st=1;
                for j=1:9
                    for k=1:18
                        GZ(j,k)=q_val(st,i)+i;  
                        st=st+1;
                    end % k
                end  % j
                surf(GX,GY,GZ);
                view(30,-15);
                hold on;
            end  %for i
            
            saveas(gcf,'Q-Mesh.jpg');
    
   disp('QAQ');
   %figure(h1);
   %title(strcat('Success at ',num2str(trial)),' trials');
