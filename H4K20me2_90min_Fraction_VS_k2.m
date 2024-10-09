% @author: Luis Guerra (luis.guerra@kcl.ac.uk)
% 
% Code used to estimate a limit-of-detection for the Set8/Suv4-20h1
% methylation cascade on nucleosome arrays with H4isoD24 (Figure S23).
% 
% Hard-coded values are taken from the analysis carried out by the
% python script "Analysis of methylation cascade on 12-mers.py"

% Initialize arrays to store results
C_90min = [];
C_delta = [];

% Hard-coded values from previous analysis (see above).

% Observed rate constants
k1_obs = 0.0016328536360927456;
k2_obs = 0.0015577031126516034;

% Resdiuals from global, nonlinear fit
R = [0, -0.01662584, -0.00984412, -0.01539186, -0.01977244, ...
        -0.00995003, -0.02504567, 0, -0.02217877, -0.02364437, ...
        -0.00923837, -0.00466675, 0.00050033, -0.00350362, 0, ...
        0.01093773, 0.00273735, 0.01341944, 0.01476809, 0.03048573, ...
        0.03908971, 0, 0.01150908, 0.0116421, 0.01183683, ...
        0.01019621, 0.00392525, 0.01294143, 0, 0.02058902, ...
        0.02556744, 0.00945439, 0.00304933, -0.00229835, 0.00395834, ...
        0, -0.00732398, 0.00172318, -0.00625488, -0.00592872, ...
        -0.0191686, -0.02216602, 0, 0.00511676, -0.00179799, ...
        0.00355503, 0.00957623, 0.00602478, 0.01210424, 0, ...
        0.00158975, -0.00192307, -0.00021602, 0.00161742, 0.00179802, ...
        -0.00045472, 0, -0.00361375, -0.00446053, -0.00716456, ...
        -0.00883937, -0.01131712, -0.01692369];

% Covariance matrix from global, nonlinear fit.
pcov = [2.83262224e-09, 1.97750034e-08;
         1.97750034e-08, 7.52319681e-07];

% Integrated rate law for fraction of H4K20me2 over time.
model = @(k,t) 1 + ((k(1) * exp(-k(2) * t) - k(2) * exp(-k(1) * t))/(k(2) - k(1)));

% Calculate the fraction of H4K20me2 at 90 min as a function of k2
% Note that the first point uses the experimentally observed value for k2.
for k2 = [k2_obs linspace(0.003,0.033,31)]
    [ypred,delta] = nlpredci(model,90,[k1_obs, k2],R,"Covariance",pcov);
    C_90min(end+1) = ypred;
    C_delta(end+1) = delta;
end

% Error bars are fixed using the precision of the experimentally observed
% k2.
delta = C_delta(1);

% Plot the results
x = [k2_obs linspace(0.003,0.033,31)];
y = C_90min;
err = delta*ones(1, length(x));
errorbar(x,y,err,".","MarkerSize",25,"MarkerFaceColor",[0.65 0.85 0.90],"LineStyle","none")
hold on
plot([0, 0.035], [C_90min(1) + delta, C_90min(1) + delta])
plot([0 0.035], [C_90min(1) - delta, C_90min(1) - delta])
xlabel('k_{2,obs} (min^{-1})')
ylabel('[H4K20me2] at 90 min')
