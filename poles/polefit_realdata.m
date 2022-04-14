% primary controls

plot_random_windows = false;








%% load in data
cap_dat = load('U238cap.mat');

% figure(1); clf
% loglog(cap_dat.x,cap_dat.y); title('Total Reconstructed \sigma'); 

% find resolve resonance range and less-dense energy grid
RRR_index = find(cap_dat.x>3.5 & cap_dat.x<140);
course_RRR_index = RRR_index(1:2:end);

capE = cap_dat.x(course_RRR_index); capXS = cap_dat.y(course_RRR_index);

% figure(2); clf
% loglog(capE,capXS); title('Course Reconstructed \sigma in RRR'); 

%% break the data up into tractable windows of ~500 energy points


ppw = 500;
FullWindows = floor(length(capE)/(ppw/2)) - 1;
Ewindows = zeros(FullWindows-1, ppw); XSwindows = zeros(FullWindows-1, ppw);

for iW = 1:FullWindows
    FirstEnergyIndex = 1+(ppw/2)*(iW-1) ;
    LastEnergyIndex = FirstEnergyIndex+ppw-1;
    Ewindows(iW,:) = capE(FirstEnergyIndex:LastEnergyIndex);
    XSwindows(iW,:) = capXS(FirstEnergyIndex:LastEnergyIndex);
end

LastEnergyIndex_LastFullWindow = 1+(ppw/2)*(iW-1) + 500-1;
LeftoverEnergyPoints = length(capE) - LastEnergyIndex_LastFullWindow;


%% plot random windows for inspection
if plot_random_windows
    show_grid_spacing = false ;
    number_of_sample_cases = 1 ;

    for iW = randi(818,1,number_of_sample_cases)
        if show_grid_spacing
            figure(iW); clf
            plot(Ewindows(iW,:), XSwindows(iW,:));hold on
            ylabel('Cross Section')
        %     figure(iW+1);clf
            yyaxis right
            plot(Ewindows(iW,1:end-1),diff(Ewindows(iW,:)))
            ylabel('Spacing in energy grid')
            xlabel('eV')
        else
            figure(iW); clf
            loglog(Ewindows(iW,:), XSwindows(iW,:));hold on
            ylabel('Cross Section')
        end
    end
end

%



%% solve window by window
plotting_baron_solve = true;
NumPeaks = 1 ; % max per window

for iW = 1:1 %FullWindows

WE = Ewindows(iW,:);
WXS = XSwindows(iW,:);

WXS = WXS(find(WE>4 & WE<4.75));
WE = WE(find(WE>4 & WE<4.75));

% create cross section function
xs_func = @(w) 0; z=WE;
for iRes = 1:NumPeaks
%     f = @(rr,irip,rp,ipsqr) -irip(iRes)./((rp(iRes)-z).^2+ipsqr) + rr(iRes).*(rp(iRes)-z)./((rp(iRes)-z).^2+ipsqr(iRes)) ;
    xs_func = @(w) xs_func(w) + -w(2+4*(iRes-1))./((w(3+4*(iRes-1))-z).^2+w(4+4*(iRes-1))) + w(1+4*(iRes-1)).*(w(3+4*(iRes-1))-z)./((w(3+4*(iRes-1))-z).^2+w(4+4*(iRes-1))) ;
end


poly_index = 4*NumPeaks+NumPeaks ;
polynomial_terms = 6;
% background_polynomial = @(w) w(poly_index+1).*z + w(poly_index+2).*z.^2 ;

xs_func = @(w) xs_func(w) + w(poly_index+1).*z + w(poly_index+2).*z.^2 + w(poly_index+3).*z.^3 + w(poly_index+4).*z.^4 + w(poly_index+5).*z.^5 + w(poly_index+6).*z.^6;


%% plot xs_func with guesses for solution to validate xs_func

% p = []; r = []; 
% Elevels = [4.4, 4.566]; widths = [.0001, 1e-2];
% height_control = [3 100];
% for iRes = 1:NumPeaks
%     p = [p, Elevels(iRes)+widths(iRes)*1i]; 
%     r = [r, height_control(iRes)*exp(3*pi/2*1i)];
% end
% ir = imag(r);
% rr = real(r);
% ip = imag(p);
% rp = real(p);
% 
% sol_parm = [];
% for iRes = 1:NumPeaks
%     sol_parm = [sol_parm rr(iRes),ir(iRes)*ip(iRes),rp(iRes),ip(iRes)^2];
% end
% sol_parm = [sol_parm, 1, 1, 3, 4];
% figure(1); clf
% loglog(WE,WXS, 'DisplayName','baron sol'); hold on
% loglog(WE,xs_func(sol_parm), 'DisplayName','baron sol')


%%
% set constraints

E_level_min = WE(1); % index is much faster than max/min funcs
E_level_max = WE(end);

parm_minvec = [-100 -100 E_level_min -100];
parm_maxvec = [100 100 E_level_max 100];

res_parm_per_window = 4*NumPeaks;
A_Lower=[diag(ones(1,res_parm_per_window)),zeros(res_parm_per_window,NumPeaks)];
A_Upper=[diag(ones(1,res_parm_per_window)),zeros(res_parm_per_window,NumPeaks)];
for jj=1:NumPeaks
    Index1=4*(jj-1); % striding function
    Index2=res_parm_per_window+jj;
    A_Lower([1+Index1,2+Index1,3+Index1,4+Index1],Index2)=-parm_minvec;
    A_Upper([1+Index1,2+Index1,3+Index1,4+Index1],Index2)=-parm_maxvec;
end

% EnergyOrder=zeros(NumPeaks-1,4*NumPeaks);
% PeakSpacing=5;
% for jj=1:(NumPeaks-1)
%     EnergyOrder(jj,3+parm_per_res*(jj-1))=-1;
%     EnergyOrder(jj,3+parm_per_res*jj)=1;
%     EnergyOrder(jj,parm_per_window+jj)=-PeakSpacing/2;
%     EnergyOrder(jj,parm_per_window+(jj+1))=-PeakSpacing/2;
% end

TotalRM_PerWindow = NumPeaks*4;
TotalParm_PerWindow=NumPeaks*(4+1)+poly_parm; % 4 + 1 binary parms for each res + 1 parameter for polynomial

% A = [A_Lower;A_Upper;EnergyOrder]; 
% SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow),zeros(1,NumPeaks-1)];
% SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow),inf(1,NumPeaks-1)];
A = [A_Lower;A_Upper]; 
SC_LowerBounds=[zeros(1,TotalRM_PerWindow),inf(1,TotalRM_PerWindow)];
SC_UpperBounds=[-inf(1,TotalRM_PerWindow),zeros(1,TotalRM_PerWindow)];
lb=[repmat(parm_minvec,1,NumPeaks), ones(1,NumPeaks)];
ub=[repmat(parm_maxvec,1,NumPeaks), ones(1,NumPeaks)];
% lb=[repmat(parm_minvec,1,NumPeaks), 1, ones(1,NumPeaks)];
% ub=[repmat(parm_maxvec,1,NumPeaks), 1, ones(1,NumPeaks)];

Options=baronset('threads',4,'PrLevel',1,'CutOff',1,'DeltaTerm',1,'EpsA',0.1,'MaxTime',10*60);
% Options=baronset('threads',4,'PrLevel',1,'CutOff',1,'MaxTime',5*60);
xtype=squeeze(char([repmat(["C","C","C","C"],1,NumPeaks), repmat(["B"],1,NumPeaks)]))';
% xtype=squeeze(char([repmat(["C","C","C","C"],1,NumPeaks), "C", repmat(["B"],1,NumPeaks)]))';

% x0 = NaN(1,4*NumPeaks+NumPeaks+polynomial_terms); %NaN(1,4*peaks+peaks);
% x0 = [sol_parm, 1 1] ; %NaN(1,4*NumPeaks+NumPeaks); %NaN(1,4*peaks+peaks);
x0 = w; 

f_obj = @(w) sum((xs_func(w)-WXS).^2) ;
% [w,fval,~,~] = baron(f_obj,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,x0, Options);
[w,fval,~,~] = baron(f_obj,[],[],[],[],[],[],[],[],[],x0, Options);


if plotting_baron_solve
    figure(1); clf
    loglog(WE,WXS, '.', 'DisplayName','Exp Data'); hold on
    plot(WE,xs_func(w), 'DisplayName','baron sol')
    legend()
%     saveas(myfig,'figure.png');
end



% end window for loop
end
