%% simple form of pole parameterization

% f = @(E,x,a) 0 ; 
% a = [10+0.5i, 15+0.5i]; % location plus width 
% x = [2*exp(3*pi/2*1i), 2*exp(3*pi/2*1i)]; % in the exponent rotation will change the type of res (scat, cap ect)
% % e should be sqrt(E), we need to fit kennys data with this
% % fit Esig(E) to Re(r/z-p) where z=sqrt(E)
% 
% for i = 1:1
%     f = @(E) f(E) + real(x(i)./(E-a(i))) ; %+ 1 + 2*3;
% end
% 
% figure(3);clf
% plot(energies, f(energies),'DisplayName','simple form'); hold on

%%
re_sample_resparm = false;
run_baron_bool = true;

NumPeaks = 5;
parm_per_res = 4;
parm_per_window = NumPeaks*parm_per_res;

WE = linspace(1, 100, 1000);
if re_sample_resparm
    Elevels = rand(1,NumPeaks).*max(WE);
    widths = rand(1,NumPeaks);
end

z=WE;

p = []; r = []; %p = 10+0.1i; r = 2*exp(3*pi/2*1i);
for iRes = 1:NumPeaks
    p = [p, Elevels(iRes)+widths(iRes)*1i];
    r = [r, 2*exp(3*pi/2*1i)];
end


f = @(c,b) 0; 
for iRes = 1:NumPeaks
%     f = @(rr,irip,rp,ipsqr) -irip(iRes)./((rp(iRes)-z).^2+ipsqr) + rr(iRes).*(rp(iRes)-z)./((rp(iRes)-z).^2+ipsqr(iRes)) ;
    f = @(w) f(w) + -w(2+4*(iRes-1))./((w(3+4*(iRes-1))-z).^2+w(4+4*(iRes-1))) + w(1+4*(iRes-1)).*(w(3+4*(iRes-1))-z)./((w(3+4*(iRes-1))-z).^2+w(4+4*(iRes-1))) ;
end

ir = imag(r);
rr = real(r);
ip = imag(p);  % careful!! I was renaming pi!!!
rp = real(p);

sol_parm = [];
for iRes = 1:NumPeaks
    sol_parm = [sol_parm rr(iRes),ir(iRes)*ip(iRes),rp(iRes),ip(iRes)^2];
end

xs = @(w) f(w);
true_xs = xs(sol_parm); 

figure(1); clf
plot(WE, true_xs,'.','DisplayName','true'); hold on

f_obj = @(w) sum((xs(w)-true_xs).^2) ;


%% solve with baron

if run_baron_bool

MinVec = [-100 -100 -100 -100];
MaxVec = [100 100 100 100];

A_Lower=[diag(ones(1,parm_per_window)),zeros(parm_per_window,NumPeaks)];
A_Upper=[diag(ones(1,parm_per_window)),zeros(parm_per_window,NumPeaks)];
for jj=1:NumPeaks
    Index1=4*(jj-1); % striding function
    Index2=parm_per_window+jj;
    A_Lower([1+Index1,2+Index1,3+Index1,4+Index1],Index2)=-MinVec;
    A_Upper([1+Index1,2+Index1,3+Index1,4+Index1],Index2)=-MaxVec;
end

EnergyOrder=zeros(NumPeaks-1,4*NumPeaks);
PeakSpacing=5;
for jj=1:(NumPeaks-1)
    EnergyOrder(jj,1+parm_per_res*(jj-1))=-1;
    EnergyOrder(jj,1+parm_per_res*jj)=1;
    EnergyOrder(jj,parm_per_window+jj)=-PeakSpacing/2;
    EnergyOrder(jj,parm_per_window+(jj+1))=-PeakSpacing/2;
end

TotalRM_PerWindow = NumPeaks*parm_per_res;
TotalParm_PerWindow=NumPeaks*(parm_per_res+1);

A = [A_Lower;A_Upper;EnergyOrder]; 
SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow),zeros(1,NumPeaks-1)];
SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow),inf(1,NumPeaks-1)];
lb=[repmat(MinVec,1,NumPeaks),ones(1,NumPeaks)];
ub=[repmat(MaxVec,1,NumPeaks),ones(1,NumPeaks)];

Options=baronset('threads',4,'PrLevel',1,'CutOff',10,'DeltaTerm',1,'EpsA',0.1,'MaxTime',5*60);
% Options=baronset('threads',4,'PrLevel',1,'CutOff',1,'MaxTime',5*60);
xtype=squeeze(char([repmat(["C","C","C","C"],1,NumPeaks),repmat(["B"],1,NumPeaks)]))';

x0 = NaN(1,4*NumPeaks+NumPeaks); %NaN(1,4*peaks+peaks);
% x0 = w; 

[w,~,~,~] = baron(f_obj,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,x0, Options);


figure(1);
plot(WE,xs(w), 'DisplayName','baron sol')
legend()

end