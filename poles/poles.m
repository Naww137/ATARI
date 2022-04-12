

energies = linspace(1, 20, 1000);

f = @(E,x,a) 0 ; 

a = [10+0.5i, 15+0.5i]; % location plus width 
x = [2*exp(3*pi/2*1i), 2*exp(3*pi/2*1i)]; % in the exponent rotation will change the type of res (scat, cap ect)
% e should be sqrt(E), we need to fit kennys data with this
% fit Esig(E) to Re(r/z-p) where z=sqrt(E)

for i = 1:1
    f = @(E) f(E) + real(x(i)./(E-a(i))) ; %+ 1 + 2*3;
end



figure(3);clf
plot(energies, f(energies),'DisplayName','simple form'); hold on

%%


z=energies;
p = 10+0.5i;
r = 2*exp(3*pi/2*1i);


f = @(c,b) 0; 
for i = 1:1
    f = @(rr,irip,rp,ipsqr) -irip(i)./((rp(i)-z).^2+ipsqr) + rr(i).*(rp(i)-z)./((rp(i)-z).^2+ipsqr(i)) ;
end

ir = imag(r);
rr = real(r);
ip = imag(p);  % careful!! I was renaming pi!!!
rp = real(p);

sol_parm = [rr,ir*ip,rp,ip^2];

xs = @(parm) f(parm(1),parm(2),parm(3),parm(4));

plot(energies, xs(sol_parm),'.','DisplayName','new form (data)')
plot(energies,xs([rr,ir*ip,rp,ip^2]), 'DisplayName','fobj(sol)');

f_obj = @(parm) sum((xs(parm)-true_xs).^2) ;


%%

peaks = 1;
parm_per_res = 4;
parm_per_window = peaks*parm_per_res;

MinVec = [-100 -100 -100 -100];
MaxVec = [100 100 100 100];

A_Lower=[diag(ones(1,parm_per_window)),zeros(parm_per_window,peaks)];
A_Upper=[diag(ones(1,parm_per_window)),zeros(parm_per_window,peaks)];
for jj=1:peaks
    Index1=4*(jj-1); % striding function
    Index2=parm_per_window+jj;
    A_Lower([1+Index1,2+Index1,3+Index1,4+Index1],Index2)=-MinVec;
    A_Upper([1+Index1,2+Index1,3+Index1,4+Index1],Index2)=-MaxVec;
end

EnergyOrder=zeros(peaks-1,4*peaks);
PeakSpacing=15;
for jj=1:(peaks-1)
    EnergyOrder(jj,1+parm_per_res*(jj-1))=-1;
    EnergyOrder(jj,1+parm_per_res*jj)=1;
    EnergyOrder(jj,parm_per_window+jj)=-PeakSpacing/2;
    EnergyOrder(jj,parm_per_window+(jj+1))=-PeakSpacing/2;
end

% A = [A_Lower;A_Upper;EnergyOrder];
A = [1 0 0 0 -100; %rr
     0 1 0 0 -100; %ir
     0 0 1 0 -100; %rp
     0 0 0 1 -100; %ip
     1 0 0 0  100;
     0 1 0 0  100;
     0 0 1 0  100;
     0 0 0 1  100;]; 

SC_LowerBounds = [0	0	0  0 Inf Inf Inf Inf] ;
SC_UpperBounds = [-Inf -Inf -Inf -Inf 0 0 0 0] ; 
lb = [-100 -100 -100 -100 0] ;
ub = [100 100 100 100 1] ; 

% Options=baronset('threads',4,'PrLevel',1,'CutOff',1,'DeltaTerm',1,'EpsA',0.1,'MaxTime',2*60);
Options=baronset('threads',4,'PrLevel',1,'MaxTime',2*60);
xtype=squeeze(char([repmat(["C","C","C","C"],1,peaks),repmat(["B"],1,peaks)]))';



% [parm,~,~,~] = baron(f_obj,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,[], Options);
x0 = NaN(1,5);
[parm,~,~,~] = baron(f_obj,[],[],[],[],[],[],[],[],[],x0, Options);


%%

plot(energies,xs(parm), 'DisplayName','baron sol');
legend();