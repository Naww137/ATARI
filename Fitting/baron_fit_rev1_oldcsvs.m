function baron_fit_rev1(pw_file, data_file)

% check type of read in data

% Load data
syndat_pw = readtable(pw_file) ;
syndat_data = readtable(data_file) ; 
% disp('Running matlab script')
% sort syndat?

% Nuclear Parameters and functions for scattering theory 

% Input parameters
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

% Functions
k=@(E) Constant*(A/(A+1))*sqrt(E);   % wave number
rho=@(E) k(E)*Ac;       % related to the center of mass momentum
P=@(E) rho(E);          % penatrability factor

%% solution vector

% Extract resonance ladder values from Syndat
Elevels = syndat_data.E';
Gc = 0.001 * syndat_data.Gg';
gn_square = 0.001 * syndat_data.gnx2';


% Number of resonances
num_res_actual = height(syndat_data);
NumPeaks = num_res_actual; % Number of resonance guesses

% Baron formatted vector of solution parameters
% Last num_res_guess values are binary switches for whether or not resonance exists
sol_w = zeros(1, 4 * NumPeaks);
for j=1:num_res_actual
    sol_w(3*(j-1)+1) = Gc(j);
    sol_w(3*(j-1)+2) = gn_square(j);
    sol_w(3*(j-1)+3) = Elevels(j);
    sol_w(3 * NumPeaks + j) = 1;
end


%% Create total cross section function

% Define energy grid
Energies = syndat_pw.E';
WE = Energies;  % Passing Energies to WE is artifact of Jordan-Noah edits

% Create function pieces
Pen = P(WE);
phi = rho(WE);
kE = k(WE);

% Create total cross section function
xs_func = @(w) 0;
for jj=1:NumPeaks % here index 1 is Gc, 2 is gn, 3 is Elevel
%     
    xs_func = @(w) xs_func(w) + (w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen).*w(2+3*(jj-1)).*2.*Pen./4./((w(3+3*(jj-1))-WE).^2 + ((w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen)./2).^2) ...
                                             .*cos(2.*phi) ...
                                         - (w(3+3*(jj-1))-WE).*(w(2+3*(jj-1)).*2.*Pen)./2./( (w(3+3*(jj-1))-WE).^2 + (( w(1+3*(jj-1))+w(2+3*(jj-1)).*2.*Pen )./2).^2 ) ...
                                             .*sin(2.*phi)   ;
end
xs_func = @(w) (4.*pig./kE.^2) .* (sin(phi).^2+xs_func(w));

n = 0.067166; 
trans_func = @(w) exp(-n*xs_func(w));

% Use syndat data instead of Jordan's noisy data
WC = syndat_pw.exp_trans';

% Plot theoretical and experimental syndat data
% figure(1); clf
% scatter(Energies, WC, '.', 'DisplayName', 'Syndat exp'); hold on
% plot(Energies, trans_func(sol_w),'o', 'DisplayName', 'Matlab theo') ; hold on
% plot(Energies, syndat_pw.theo_trans, 'DisplayName', 'Syndat theo')
% legend()


%% Set Baron options

fun_robust1=@(w) sum((trans_func(w)-WC).^2);

% insert min/max of Gc, gn_square, and energy 
MinVec = [0 0 min(Energies)];
MaxVec = [max(Gc)*10 max(gn_square)*10 max(Energies)];

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
PeakSpacing=2;
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
Options=baronset('threads',8,'PrLevel',0,'MaxTime',2*60);
xtype=squeeze(char([repmat(["C","C","C"],1,NumPeaks),repmat(["B"],1,NumPeaks)]))';

w0 = []; % optional inital guess


%% Baron solve

[w1,fval,ef,info]=baron(fun_robust1,A,SC_LowerBounds,SC_UpperBounds,lb,ub,[],[],[],xtype,w0, Options); % run baron

%% Plot results

% figure(1); clf
% scatter(Energies, WC, '.', 'DisplayName', 'Syndat exp'); hold on
% plot(Energies, trans_func(sol_w), 'DisplayName', 'Matlab theo') ; hold on
% plot(Energies, syndat_pw.theo_trans, 'DisplayName', 'Syndat theo'); hold on
% plot(Energies, trans_func(w1), 'DisplayName','Baron sol');
% legend()
% 
% fprintf('SE solution: %f\n',fun_robust1(sol_w))
% fprintf('SE Baron: %f\n', fval)

%% Write out results

% write estimated parameters
parameter_matrix = reshape(w1(1:NumPeaks*3),3,[])' ;
E = parameter_matrix(:,3) ; 
Gg = parameter_matrix(:,1)   *1e3 ; 
gnx2 = parameter_matrix(:,2) *1e3 ; 

pathparts = strsplit(data_file,'/');
data_dir = pathparts(1:end-1) ; 
split_filename = strsplit(string(pathparts(end)), '_');
split_filename(1) = "fit" ;
new_filename= join(split_filename, '_') ;
new_data_file = join([data_dir,new_filename],'/') ;

writetable(table(E, Gg, gnx2), new_data_file) ;

% write estimated pw transmission
E = Energies' ;
est_trans = trans_func(sol_w)' ;

pathparts = strsplit(pw_file,'/');
data_dir = pathparts(1:end-1) ; 
split_filename = strsplit(string(pathparts(end)), '_');
split_filename(1) = "fit" ;
new_filename= join(split_filename, '_') ;
new_pw_file = join([data_dir,new_filename],'/') ;

writetable(table(E, est_trans), new_pw_file) ;



%% return values 
% SE = fval; 
% sol_SE = fun_robust1(sol_w) ;
% 
% fprintf('SE = %d\n', SE)
% fprintf('sol_SE = %d\n', sol_SE)

end


