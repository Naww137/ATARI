function [w, SE, fval] = run_baron(xs_func, NumPeaks, WC, WE, run_baron_bool, initial_vec, Options)

fun_robust=@(w) sum((xs_func(w)-WC).^2);
SE = fun_robust;

% plot(WE,xs_func(true_w(1,:)))
w = [];
if run_baron_bool
    % constraints
    % min/max vec must align with the order of parameters
    % MinVec = [min(WE) min(true_w(:,2)) min(true_w(:,3))];
    % MaxVec = [max(WE) max(true_w(:,2)) max(true_w(:,3))];
    MinVec = [min(WE) 0 0];
    MaxVec = [max(WE) 1 100];
    
    RM_PerPeak = 3 ;
    
    % the following just structures the min/max vecs above for baron to handle
    % also adds a binary variables for each peak
    TotalRM_PerWindow = NumPeaks*RM_PerPeak;
    TotalParm_PerWindow=NumPeaks*(RM_PerPeak+1);
    
    A_Lower=[diag(ones(1,TotalRM_PerWindow)),zeros(TotalRM_PerWindow,NumPeaks)];
    A_Upper=[diag(ones(1,TotalRM_PerWindow)),zeros(TotalRM_PerWindow,NumPeaks)];
    for jj=1:NumPeaks
        Index1=3*(jj-1); % striding function
        Index2=TotalRM_PerWindow+jj;
        A_Lower([1+Index1,2+Index1,3+Index1],Index2)=-MinVec;
        A_Upper([1+Index1,2+Index1,3+Index1],Index2)=-MaxVec;
    end
    
    EnergyOrder=zeros(NumPeaks-1,4*NumPeaks);
    PeakSpacing=15;
    for jj=1:(NumPeaks-1)
        EnergyOrder(jj,1+RM_PerPeak*(jj-1))=-1;
        EnergyOrder(jj,1+RM_PerPeak*jj)=1;
        EnergyOrder(jj,TotalRM_PerWindow+jj)=-PeakSpacing/2;
        EnergyOrder(jj,TotalRM_PerWindow+(jj+1))=-PeakSpacing/2;
    end
    
    A = [A_Lower;A_Upper;EnergyOrder];
    SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow),zeros(1,NumPeaks-1)];
    SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow),inf(1,NumPeaks-1)];
    lb=zeros(1,TotalParm_PerWindow);
    ub=[repmat(MaxVec,1,NumPeaks),ones(1,NumPeaks)];
    
    
    % cutoff 10 is very important, also EpsA 100 makes it run much quicker 
%     Options=baronset('threads',8,'PrLevel',1,'DeltaTerm',1,'EpsA',0.1,'MaxTime',5*60);
    xtype=squeeze(char([repmat(["C","C","C"],1,NumPeaks),repmat(["B"],1,NumPeaks)]))';
    
    [w,fval,~,~]=baron(fun_robust,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,initial_vec, Options);

end


end