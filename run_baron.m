function [w, xs_func, SE] = run_baron(NumPeaks, WC, WE, run_baron, initial_vec)

% Nuclear Parameters
A=62.929599;
Constant=0.002197; 
Ac=0.67; 
I=1.5; 
ii=0.5; 
l=0;  
s=I-ii; 
J=l+s;  %
g=(2*J+1)/( (2*ii+1)*(2*I+1) );   
pig=pi*g;

%l=0   or s-wave spin group energy dependent functions of the wave number
k=@(E) Constant*(A/(A+1))*sqrt(E);  
rho=@(E) k(E)*Ac;    
P=@(E) rho(E); % not using right now

fun_robust = @(w) 0;
for jj=1:NumPeaks
    fun_robust=@(w) fun_robust(w)+( (w(3+3*(jj-1)).*w(2+3*(jj-1)))  ./ ( (WE-w(1+3*(jj-1))).^2  + ((w(3+3*(jj-1))+w(2+3*(jj-1)))./2).^2 ) );
end

fun_robust = @(w) ((pig)./k(WE).^2).*fun_robust(w) ;
xs_func = @(w) fun_robust(w) ;
fun_robust=@(w) sum((fun_robust(w)-WC).^2);
SE = fun_robust;

% plot(WE,xs_func(true_w(1,:)))
w = [];
if run_baron
    % constraints
    % min/max vec must align with the order of parameters
    % MinVec = [min(WE) min(true_w(:,2)) min(true_w(:,3))];
    % MaxVec = [max(WE) max(true_w(:,2)) max(true_w(:,3))];
    MinVec = [min(WE) 0 0];
    MaxVec = [max(WE) 1 50];
    
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
        EnergyOrder(jj,RM_PerPeak+RM_PerPeak*(jj-1))=-1;
        EnergyOrder(jj,RM_PerPeak+RM_PerPeak*jj)=1;
        EnergyOrder(jj,TotalRM_PerWindow+jj)=-PeakSpacing/2;
        EnergyOrder(jj,TotalRM_PerWindow+(jj+1))=-PeakSpacing/2;
    end
    
    A = [A_Lower;A_Upper;EnergyOrder];
    SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow),zeros(1,NumPeaks-1)];
    SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow),inf(1,NumPeaks-1)];
    lb=zeros(1,TotalParm_PerWindow);
    ub=[repmat(MaxVec,1,NumPeaks),ones(1,NumPeaks)];
    
    
    % cutoff 10 is very important, also EpsA 100 makes it run much quicker 
    Options=baronset('threads',8,'PrLevel',0,'CutOff',10,'DeltaTerm',1,'EpsA',100,'MaxTime',2*60);
%     Options=baronset('threads',8,'PrLevel',1,'DeltaTerm',1,'EpsA',0.1,'MaxTime',5*60);
    xtype=squeeze(char([repmat(["C","C","C"],1,NumPeaks),repmat(["B"],1,NumPeaks)]))';
    
    [w,~,~,~]=baron(fun_robust,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,initial_vec, Options);

end


end