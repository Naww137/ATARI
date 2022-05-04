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
plot_local = true;
plot_cluster = false;
poly_term = false;

NumPeaks = 1;




WE = linspace(3.5, 5.2, 300);
if re_sample_resparm
    Elevels = rand(1,NumPeaks).*max(WE);
    widths = rand(1,NumPeaks);

% Elevels = [1.7]; widths = [1e-5];
p = []; r = []; %p = 10+0.1i; r = 2*exp(3*pi/2*1i);
for iRes = 1:NumPeaks
    p = [p, Elevels(iRes)+widths(iRes)*1i]; 
    r = [r, 2*exp(3*pi/2*1i)];
end

ir = imag(r);
rr = real(r);
ip = imag(p);
rp = real(p);

sol_parm = [];
for iRes = 1:NumPeaks
    sol_parm = [sol_parm rr(iRes),ir(iRes)*ip(iRes),rp(iRes),ip(iRes)^2];
end
end

% add polynomial term
% poly_index = 4*NumPeaks+NumPeaks ;
poly_index = 4*NumPeaks;

% background_polynomial = @(w) w(poly_index+1).*z + w(poly_index+2).*z.^2 ;
% could use coef of legendre polynomial to simplify this polynomial term [-1,1]
% xs_func = @(w) xs_func(w) ;
% xs = @(w) f(w) + 10 ; %w(poly_index+1).*z + w(poly_index+2).*z.^2 + w(poly_index+3).*z.^3 + w(poly_index+4).*z.^4 + w(poly_index+5).*z.^5 + w(poly_index+6).*z.^6;
% sol_parm = [sol_parm rand(1,polynomial_terms).*1e-10];
polynomial_terms=3;
parm_per_res = 4;
parm_per_window = NumPeaks*parm_per_res+NumPeaks+polynomial_terms;




true_xs_func = xs_pole(NumPeaks,WE); 
if poly_term
    z = WE;
    true_xs_func = @(w) true_xs_func(w) + w(poly_index+1).*z + w(poly_index+2).*z.^2 + w(poly_index+3).*z.^3 ;
    if re_sample_resparm
        sol_parm = [sol_parm, -782.996974515679	366.628693805771	-42.6132398273227];
    else
        sol_parm = [-0.0572641867005426	-1.81466913738840	4.56853649988364	3.25249651413280e-06 -0.0572641867005426	-1.81466913738840	4.2	3.25249651413280e-06 -782.996974515679	366.628693805771	-42.6132398273227];
    end
else
    sol_parm = [-0.0572641867005426	-1.81466913738840	4.56853649988364	3.25249651413280e-06 -0.0572641867005426	-1.81466913738840	4.2 3.25249651413280e-06];
end
true_xs = true_xs_func(sol_parm);

% plot(WE, true_xs,'.','DisplayName','true'); hold on
% if plotting
%     if running_on_cluster
%         myfig = figure('Visible', 'off');
%         set(myfig,'Visible', 'off');
%     end
%     figure(1); clf
%     loglog(WE, true_xs,'.','DisplayName','true'); hold on
% end

% true_xs = (true_xs-min(true_xs))./max(true_xs);
% WE = (WE-min(WE))./max(WE);

% figure(2); clf
% plot(WE,true_xs, '.')

xs_func = xs_pole(NumPeaks,WE);
if poly_term
    z = WE;
    polynomial_terms = 3;
    xs_func = @(w) xs_func(w) + w(poly_index+1).*z + w(poly_index+2).*z.^2 + w(poly_index+3).*z.^3 ;
else 
    polynomial_terms = 0;
end
f_obj = @(w) sum((xs_func(w)-true_xs).^2) ;


%% solve with baron

if run_baron_bool

MinVec = [-100 -100 0 -100];
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
PeakSpacing=1;
for jj=1:(NumPeaks-1)
    EnergyOrder(jj,3+parm_per_res*(jj-1))=-1;
    EnergyOrder(jj,3+parm_per_res*jj)=1;
    EnergyOrder(jj,parm_per_window+jj)=-PeakSpacing/2;
    EnergyOrder(jj,parm_per_window+(jj+1))=-PeakSpacing/2;
end

TotalRM_PerWindow = NumPeaks*parm_per_res;
TotalParm_PerWindow=NumPeaks*(parm_per_res+1);

% A = [A_Lower;A_Upper;EnergyOrder]; 
% SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow),zeros(1,NumPeaks-1)];
% SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow),inf(1,NumPeaks-1)];
A = [A_Lower;A_Upper]; 
SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow)];
SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow)];
lb=[repmat(MinVec,1,NumPeaks),ones(1,NumPeaks)];
ub=[repmat(MaxVec,1,NumPeaks),ones(1,NumPeaks)];

% Options=baronset('threads',4,'PrLevel',1,'CutOff',1,'DeltaTerm',1,'EpsA',0.1,'MaxTime',2*60);
% Options=baronset('threads',4,'PrLevel',1,'CutOff',1,'MaxTime',5*60);
Options=baronset('threads',4,'PrLevel',1,'MaxTime',2*60);
xtype=squeeze(char([repmat(["C","C","C","C"],1,NumPeaks),repmat(["B"],1,NumPeaks)]))';


x0 = NaN(1,4*NumPeaks+polynomial_terms); %NaN(1,4*peaks+peaks);
% x0 = w; 
% x0 = sol_parm;

% [w,~,~,~] = baron(f_obj,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,x0, Options);
[w,~,~,~] = baron(f_obj,[],[],[],[],[],[],[],[],[],x0, Options);



if plot_cluster
    myfig = figure('Visible', 'off'); hold on
    set(gca, 'YScale', 'log')
    set(gca, 'XScale', 'log')
    loglog(WE, true_xs,'.','DisplayName','true'); hold on
    loglog(WE,xs_func(w), 'DisplayName','baron sol')
    legend()
    saveas(myfig,'figure.png');
%         set(myfig,'Visible', 'off');
end
if plot_local
    figure(2);clf
    loglog(WE, true_xs,'.','DisplayName','true'); hold on
    loglog(WE,xs_func(w), 'DisplayName','baron sol')
end

end