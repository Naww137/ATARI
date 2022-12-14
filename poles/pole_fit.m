
capdat = load('u238cap.mat');

capE = capdat.x;
capxs = capdat.y;

% find resolved resonance range - initial, total window
upper_lim_RRR = find(round(capE)==140);
lower_lim_RRR = find(round(capE)==3);
capE = capE(lower_lim_RRR:upper_lim_RRR);
capxs = capxs(lower_lim_RRR:upper_lim_RRR);
length(capE)

capE = capE(1:1e3);
capxs = capxs(1:1e3);
% capE = capE(1:1e4);
% capxs = capxs(1:1e4);
% capE = capE(5.4e5:5.45e5);
% capxs = capxs(5.4e5:5.45e5);

plotting = true;

if plotting
    figure(1); clf
    loglog(capE,capxs, '.'); hold on
    for i = 1:1
        xline(capE((500*i)))
    end
    xlabel('eV')
    legend('\sigma','500 energy points')
end

%%

n = 5;
E_red = arrayfun(@(i) mean(capE(i:i+n-1)),1:n:length(capE)-n+1);
xs_red = arrayfun(@(i) mean(capxs(i:i+n-1)),1:n:length(capxs)-n+1);

if plotting
    figure(2); clf
    loglog(E_red,xs_red, '.'); hold on
%     for i = 1:1
%         xline(capE((500*i)))
%     end
    xlabel('eV')
%     legend('\sigma','500 energy points')
end

%%
% SOLVER OPTIONS
run_baron_bool = true ;
iterate_baron = false;
normalize_range = false;
constraints = true;
give_solution = false;
fudge = 0.9;

% BARON RUNTIME OPTIONS
maximum_total_time = 10*60; % 2*60*60; %
absolute_tolerance = 0.01; % absolute tolerance should be lower if transforming to [0,1]
print_out = 1;

initial_vec = [];
options_first_run = baronset('threads',8,'PrLevel',print_out,'EpsA',absolute_tolerance,'MaxTime',maximum_total_time);

NumPeaks = 5;

poly_index = 4*NumPeaks;
poly_term = 3 ;
z=E_red;
xs_func = xs_pole(NumPeaks,WE);
if poly_term
    z = WE;
    polynomial_terms = 3;
    xs_func = @(w) xs_func(w) + w(poly_index+1).*z + w(poly_index+2).*z.^2 + w(poly_index+3).*z.^3 ;
else 
    polynomial_terms = 0;
end


WE = E_red;
true_xs = xs_red; 
initial_vec = [0 0 2.6 0, 0 0 3.2 0, 0 0 3.37 0, 0 0 4.37 0, 0 0 0 0, 0 0 0];
run_pswarm = true;

%%

if run_pswarm

    fobj.ObjFunction = @(w) sum((xs_func(w)-true_xs).^2);

    MinVec = [-100 -100 WE(1) -100];
    MaxVec = [ 100  100 WE(end) 100];

        RM_PerPeak = 4 ;
        TotalRM_PerWindow = NumPeaks*RM_PerPeak;
        TotalParm_PerWindow=NumPeaks*(RM_PerPeak);
        
        minimum = []; maximum = [];
        for jj=1:NumPeaks
            minimum = [minimum; MinVec'];
            maximum = [maximum; MaxVec'];
        end
        if poly_term
            minimum = [minimum; -1000; -1000; -1000];
            maximum = [maximum; 1000; 1000; 1000];
            fobj.Variables = TotalParm_PerWindow + 3 ;
        else
            fobj.Variables = TotalParm_PerWindow ;
        end
        

        fobj.LB = minimum;
        fobj.UB = maximum;
        
        InitialPopulation = [];
        if isempty(initial_vec)
        else
            InitialPopulation(1).x = initial_vec;
            % InitialPopulation(2).x = [2,3]; second guess
        end

        % could add other constraints
        Options = PSwarm('defaults') ;
        Options.MaxObj = 1e6;
        Options.MaxIter = Options.MaxObj ;
        Options.CPTolerance = 1e-7;
        Options.DegTolerance = 1e-5;
        Options.IPrint = 1000;
        Options.SearchType = 1 ;

        [w, fval, RunData] = PSwarm(fobj,InitialPopulation,Options);


plot_local = true;
if plot_local
    figure(2);clf
    plot(WE, true_xs,'.','DisplayName','true'); hold on
    plot(WE,xs_func(w), 'DisplayName','baron sol')
end

end

%%
% set constraints
% 
% E_level_min = WE(1); % index is much faster than max/min funcs
% E_level_max = WE(end);
% 
% parm_minvec = [-100 -100 E_level_min -100];
% parm_maxvec = [100 100 E_level_max 100];
% 
% res_parm_per_window = 4*NumPeaks;
% A_Lower=[diag(ones(1,res_parm_per_window)),zeros(res_parm_per_window,NumPeaks)];
% A_Upper=[diag(ones(1,res_parm_per_window)),zeros(res_parm_per_window,NumPeaks)];
% for jj=1:NumPeaks
%     Index1=4*(jj-1); % striding function
%     Index2=res_parm_per_window+jj;
%     A_Lower([1+Index1,2+Index1,3+Index1,4+Index1],Index2)=-parm_minvec;
%     A_Upper([1+Index1,2+Index1,3+Index1,4+Index1],Index2)=-parm_maxvec;
% end
% 
% % EnergyOrder=zeros(NumPeaks-1,4*NumPeaks);
% % PeakSpacing=5;
% % for jj=1:(NumPeaks-1)
% %     EnergyOrder(jj,3+parm_per_res*(jj-1))=-1;
% %     EnergyOrder(jj,3+parm_per_res*jj)=1;
% %     EnergyOrder(jj,parm_per_window+jj)=-PeakSpacing/2;
% %     EnergyOrder(jj,parm_per_window+(jj+1))=-PeakSpacing/2;
% % end
% 
% TotalRM_PerWindow = NumPeaks*4;
% TotalParm_PerWindow=NumPeaks*(4+1)+polynomial_terms; % 4 + 1 binary parms for each res + 1 parameter for polynomial
% 
% % A = [A_Lower;A_Upper;EnergyOrder]; 
% % SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow),zeros(1,NumPeaks-1)];
% % SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow),inf(1,NumPeaks-1)];
% A = [A_Lower;A_Upper]; 
% SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow)];
% SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow)];
% lb=[repmat(parm_minvec,1,NumPeaks), ones(1,NumPeaks)];
% ub=[repmat(parm_maxvec,1,NumPeaks), ones(1,NumPeaks)];
% % lb=[repmat(parm_minvec,1,NumPeaks), 1, ones(1,NumPeaks)];
% % ub=[repmat(parm_maxvec,1,NumPeaks), 1, ones(1,NumPeaks)];
% 
% % Options=baronset('threads',4,'PrLevel',1,'CutOff',1,'DeltaTerm',1,'EpsA',0.1,'MaxTime',2*60);
% Options=baronset('threads',8,'PrLevel',1,'MaxTime',2*60);
% xtype=squeeze(char([repmat(["C","C","C","C"],1,NumPeaks), repmat(["B"],1,NumPeaks)]))';
% % xtype=squeeze(char([repmat(["C","C","C","C"],1,NumPeaks), "C", repmat(["B"],1,NumPeaks)]))';
% 
% x0 = NaN(1,4*NumPeaks+NumPeaks+polynomial_terms); %NaN(1,4*peaks+peaks);
% % x0 = [sol_parm, 1 1] ; %NaN(1,4*NumPeaks+NumPeaks); %NaN(1,4*peaks+peaks);
% % x0 = w; 
% 
% f_obj = @(w) sum((xs_function(w)-xs_red).^2) ;
% [w,fval,~,~] = baron(f_obj,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,x0, Options);
% % [w,fval,~,~] = baron(f_obj,[],[],[],[],[],[],[],[],[],x0, Options);
% 
% 
% if plotting_baron_solve
%     figure(1); clf
%     loglog(WE,WXS, '.', 'DisplayName','Exp Data'); hold on
%     plot(WE,xs_func(w), 'DisplayName','baron sol')
%     legend()
% %     saveas(myfig,'figure.png');
% end
% 
% %% window selection
% ppw = 500;
% NumWindows=ceil(length(capE)/(ppw/2)) ;
% EndWindow_pts = int8(length(capE)/(ppw/2)-(NumWindows-1))*(ppw/2);
% 
% WindowCrossSection=zeros(NumWindows-1,ppw);
% % WindowCrossSection_std=zeros(NumWindows,ppw);
% WindowEnergies=zeros(NumWindows-1,ppw) ;
% 
% Energy_start = min(capE);
% 
% for iW = 1:NumWindows-2
% 
% %     if iW == 1
% %         shift = 0;
% %     else
% %         shift = (ppw)/2;
% %     end
%         
%     FirstEnergyIndex = 1+(ppw/2)*(iW-1) ;
%     LastEnergyIndex = FirstEnergyIndex+ppw-1;
% 
%     WindowEnergies(iW,:)= capE(FirstEnergyIndex:LastEnergyIndex);
%     WindowCrossSection(iW,:)=capxs(FirstEnergyIndex:LastEnergyIndex);
% %     WindowCrossSection_std(iW,:)=capxs(FirstEnergyIndex:LastEnergyIndex);
% 
% end
% 
% figure(2); clf
% for iW = 1:NumWindows-1
%     loglog(WindowEnergies(iW,:),WindowCrossSection(iW,:)); hold on
%     
% end
% 
% % for iW = 1:3
% %     xline(min(WindowEnergies(iW,:))); xline(max(WindowEnergies(iW,:)))
% % end

