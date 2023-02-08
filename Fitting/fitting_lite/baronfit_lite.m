%% Nuclear Parameters and functions for scattering theory 

A=180.948434; %Cu-63, number from cu63 input txt file
Constant=0.00219680781008; %sqrt(2Mn)/hbar; units of (10^(-12) cm sqrt(eV)^-1)
Ac=0.81271; % scattering radius 6.7 fermi expressed as 10^-12 cm
I=3.5; % target angular Momentum
ii=0.5; % incident angular momentum
l=0;   % l=0 or s-wave spin group

s=I-ii; %
J=l+s;  % 
g=(2*J+1)/( (2*ii+1)*(2*I+1) );   % spin statistical factor g sub(j alpha)

pig=pi*g;

%l=0   or s-wave spin group energy dependent functions of the wave number
k=@(E) Constant*(A/(A+1))*sqrt(E);   % wave number
rho=@(E) k(E)*Ac;       % related to the center of mass momentum
P=@(E) rho(E);          % penatrability factor
% phi = @(E) rho(E) ; 

%% solution vector

% These are coefficients for equation at bottom of page 32 in SAMMY manual 
% (Rich Moore approximation for single level)
Gc = [0.4811, 0.5362] ;
Gn = [.113, .847];
Elevels = [1.94707086e+03, 2.054739864e+03]; %linspace(3000,4000,1) ;
gn_square = Gn./2./P([Elevels(1), Elevels(2)]) ;

% Baron formatted vector of solution parameters
% Last three values are binary switches for whether or not resonance exists
% sol_w = [Gc(1) gn_square(1) Elevels(1) Gc(2) gn_square(2) Elevels(2) 0 0 0 1 1 0] ;
sol_w = [Gc(1) gn_square(1) Elevels(1) 0 0 0 0 0 0 1 0 0] ;
% sol_w = [0 0 0 0 0 0 0 0 0 0 0 0] ;

% DEFINE ENERGY GRID
Energies = linspace(1900, 2100, 200);
WE = Energies;

% number of peaks baron will solve for
NumPeaks = 3;


% create cross section function(s)

% total
Pen = P(WE);
phi = rho(WE);
kE = k(WE) ;
xs_func = @(w) 0;
for jj=1:NumPeaks % here index 1 is Gc, 2 is gn, 3 is Elevel
    
    xs_func = @(w) xs_func(w) + (w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen).*w(2+3*(jj-1)).*2.*Pen./4./((w(3+3*(jj-1))-WE).^2 + ((w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen)./2).^2) ...
                                             .*cos(2.*phi) ...
                                         - (w(3+3*(jj-1))-WE).*(w(2+3*(jj-1)).*2.*Pen)./2./( (w(3+3*(jj-1))-WE).^2 + (( w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen )./2).^2 ) ...
                                             .*sin(2.*phi)   ;
end
xs_func = @(w) (4.*pig./kE.^2) .* (sin(phi).^2+xs_func(w));


% Gg = [Gc(1), 0];
% Gn = [gn_square(1).*2.*Pen; 0.*2.*Pen];
% El = [Elevels(1), 0];
% xs=0;
% for jj=1:2 % here index 1 is Gc, 2 is gn, 3 is Elevel
% 
%     xs = xs + (Gg(jj)+Gn(jj,:)).*Gn(jj,:)./4./((El(jj)-WE).^2 + ((Gg(jj)+Gn(jj,:))./2).^2) ...
%                                              .*cos(2.*phi) ...
%                                          - (El(jj)-WE).*( Gn(jj,:) )./2./( (El(jj)-WE).^2 + (( Gg(jj)+Gn(jj,:) )./2).^2 ) ...
%                                              .*sin(2.*phi)   ;
% end
% xs = (4.*pig./kE.^2) .* (sin(phi).^2+xs); 

% Gg = w(1+3*(jj-1)) ;
% Gn = w(2+3*(jj-1)).*2.*Pen ;
% El = w(3+3*(jj-1));

n = 0.067166; 
trans_func = @(w) exp(-n*xs_func(w));

% capture
% TwiceP=2*P(WE);
% xs_func1 = @(w) 0;
% for jj=1:NumPeaks  % here index 1 is Gc, 2 is gn, 3 is Elevel
%     xs_func1=@(w) xs_func1(w)+( (w(2+3*(jj-1)).*w(1+3*(jj-1)))  ./ ( (WE-w(3+3*(jj-1))).^2  +( (TwiceP.*w(2+3*(jj-1))+w(1+3*(jj-1)))  ./2).^2) );
% end
% xs_func1 = @(w) (((TwiceP*pig)./k(WE).^2).*xs_func1(w)) ;


% calculate true cross section or transmission model
% true = trans_func(sol_w); 
true = xs_func(sol_w);

% compare to a sammy calculation
sammy = readmatrix('/Users/noahwalton/Library/Mobile Documents/com~apple~CloudDocs/Research Projects/Resonance Fitting/ATARI_workspace/sammy/SAMMY.LST', 'FileType','text');
syndat = readmatrix('/Users/noahwalton/Library/Mobile Documents/com~apple~CloudDocs/Research Projects/Resonance Fitting/ATARI_workspace/sammy/syndat', 'FileType','text');

% add noise and plot synthetic data
% if fitting transmission, noise parameters must be decreased
% a=15;
% b=100;
a=250;
b=10;
Noisy_CrossSection_std=zeros(1,length(Energies));
Noisy_CrossSection=zeros(1,length(Energies));
for jj=1:1
    [Noisy_CrossSection_std(jj,:),Noisy_CrossSection(jj,:)]=Noise(true,a,b);
end

WC = Noisy_CrossSection; 

figure(1); clf
scatter(Energies, WC, '.', 'DisplayName', 'Exp'); hold on
plot(Energies, true, 'DisplayName', 'Matlab') ; hold on
% plot(syndat(:,1), syndat(:,2), '.', 'DisplayName', 'Syndat')
% plot(sammy(:,1), sammy(:,4), 'DisplayName', 'Sammy')
legend()


%% solve baron 1 Original implementation - Works!

% fun_robust1=@(w) sum((xs_func(w)-WC).^2);
fun_robust1=@(w) sum((trans_func(w)-WC).^2);

% insert min/max of Gc, gn_square, and energy 
MinVec = [0 0 min(Energies)];
MaxVec = [25 10 max(Energies)];

% automated setup of baron inputs
RM_PerPeak = 3 ;
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

% baron runtime options
Options=baronset('threads',8,'PrLevel',0,'MaxTime',30, 'LPSol', 8);
%%
xtype=squeeze(char([repmat(["C","C","C"],1,NumPeaks),repmat(["B"],1,NumPeaks)]))';

w0 = []; % optional inital guess
[w1,fval,ef,info]=baron(fun_robust1,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,w0, Options); % run baron

%%

figure(1); clf
plot(Energies,WC, '.', 'DisplayName', 'Synth Exp Data'); hold on
% plot(Energies, xs_func(w1), 'DisplayName','Baron Sol1');
% plot(Energies, xs_func(sol_w), 'DisplayName','Sol Vec');
plot(Energies, trans_func(w1), 'DisplayName','Baron Sol1');
plot(Energies, trans_func(sol_w), 'DisplayName','Sol Vec');
legend()

fprintf('SE solution: %f\n',fun_robust1(sol_w))
fprintf('SE Baron: %f\n', fval)

%%


% ========
% function to add noise to synthetic true cross section
% ========
function [std,New_CrossSection]=Noise(CrossSection,a,b)
Trans_sigma=a*CrossSection+b;
NoisyTrans_sigma=normrnd(Trans_sigma,sqrt(Trans_sigma));
New_CrossSection=(NoisyTrans_sigma-b)/a; %
std=(sqrt(Trans_sigma))/a;
end