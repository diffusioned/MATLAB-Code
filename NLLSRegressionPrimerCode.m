function NLLSRegressionPrimerCode()

x = 0:0.1:10;
beta1 = 0.5;
beta2 = 5;
NumParams = 2;
stdnoise = 0.1;

y = exp(-beta1.*x).*cos(beta2.*x);
ynoise = y;
for i = 1:length(x)
    ynoise(i) = y(i) + stdnoise*randn(1);
end

% If you want to play around with the same signal every time
% save('C:\YourDirectory\RSSDemo.mat','ynoise');
% After running the above code once use this for future tests
% load('C:\YourDirectory\RSSDemo.mat','ynoise');

% Line plot
figure('Position', [50,50,600,600]);
plot(x,y,'b','LineWidth',1.5);hold on; % Original data
plot(x,ynoise,'r','LineWidth',2); % Simulated noisy data
xlabel('x','FontWeight','bold','fontsize', 14); % x axis label
ylabel('y','FontWeight','bold','fontsize', 14); % y axis label
set(gca,'FontSize',12,'FontWeight','bold'); % Fix axis label font size

% NLLS Fit
StartGuess = [1,1];
FitOptions = optimset('Display', 'off', 'Algorithm', 'levenberg-marquardt','Jacobian','on'); % 'Jacobian','on'
[RetParams,RSS,Residuals,ExitFlag,Output,RetLamb,RetJacob] = lsqcurvefit(@DecayCosineWithJac, StartGuess, x, ynoise, [], [], FitOptions);

% Residual Histogram
figure('Position', [50,50,600,600]);
histogram(Residuals,25); xlim([-0.4 0.4]);
xlabel('Residual Distance','FontWeight','bold','fontsize', 14); % x axis label
ylabel('Count','FontWeight','bold','fontsize', 14); % y axis label
set(gca,'FontSize',12,'FontWeight','bold'); % Fix axis label font size

% Calculate the RSS values in the neighborhood of the parameter values
Beta1Range = 0:0.05:7; % 5 for maximum in the first plot
Beta2Range = 0:0.1:10;
RSSForParams = zeros(length(Beta1Range),length(Beta2Range));

for i = 1:length(Beta1Range)
    for j = 1:length(Beta2Range)
        Params = [Beta1Range(i), Beta2Range(j)];
        RSSForParams(i,j) = CompareRSSOfTwoSignals(DecayCosine(Params,x),ynoise);
    end
end

% Plot a 3D contour of the RSS values
figure('Position', [50,50,700,600]);
% whitebg('k'); % if you want to set the background to black for contrast
colormap('jet'); % change the default color map for contrast
contour3(Beta2Range,Beta1Range,squeeze(RSSForParams(:,:)),0:0.025:20); % 3D contour plot
zlim([0 20]); % Set the z dimension range to match the contour plot values
xlabel('\beta2','FontWeight','bold','fontsize', 14); % x axis label
ylabel('\beta1','FontWeight','bold','fontsize', 14); % y axis label
zlabel('RSS Value','FontWeight','bold','fontsize', 14); % z axis label
set(gca,'FontSize',12,'FontWeight','bold'); % Fix axis label font size
HandleC =  colorbar('location', 'EastOutside');
HandleCLabel = ylabel(HandleC,'RSS','FontSize',14,'FontWeight','bold');
% Optional code to rotate the colorbar label 180 degrees
drawnow; % MUST STICK THIS HERE TO GET THE NEXT PARTS TO WORK
LabelPos = HandleCLabel.Position;
HandleCLabel.Rotation = -90;
HandleCLabel.Position = [LabelPos(1) + 1.1, LabelPos(2), LabelPos(3)];

% Plot the path that the algorithm takes to find the minimum value
% My results with my data.  I explicitly copied the values from the MATLAB output window.
% Your values will be different.
ParamPosArray = [[1.0000,    1.0000]; [1.0000,    1.0000]; [6.4772,    2.3668]; ...
%  [ -13.4776,   34.5894];... % Note, these are the values I commented out
%    [-3.5441,   17.8751];... % Another algorithm might do better?
    [4.6130,    4.7610];...
%    [-1.7828,    5.3593];...
    [3.4910,    4.7619]; [1.7620,    4.9382];...
%    [-1.0979,    5.0401];...
%    [-0.2964,    5.0234];...
    [1.2206,    4.9668]; [0.6158,    4.9991]; [0.4383,    5.0030]; [0.4885,    5.0016];...
    [0.4849,    5.0016]; [0.4855,    5.0016]; [0.4854,    5.0016]];

% Note: When the user-provided Jacobian is added, the algorithm takes the same path, but the number
% of interal iterations is reduced.

% Plot the 2D contour background
figure('Position', [50,50,700,600]);
% whitebg('k'); % if you want to set the background to black for contrast
colormap('jet'); % change the default color map for contrast
contour(Beta2Range,Beta1Range,squeeze(RSSForParams(:,:)),0:0.01:20); % 3D contour plot
xlabel('\beta2','FontWeight','bold','fontsize', 14); % x axis label
ylabel('\beta1','FontWeight','bold','fontsize', 14); % y axis label
set(gca,'FontSize',12,'FontWeight','bold'); % Fix axis label font size
HandleC =  colorbar('location', 'EastOutside');
HandleCLabel = ylabel(HandleC,'RSS','FontSize',14,'FontWeight','bold');
% Optional code to rotate the colorbar label 180 degrees
drawnow; % MUST STICK THIS HERE TO GET THE NEXT PARTS TO WORK
LabelPos = HandleCLabel.Position;
HandleCLabel.Rotation = -90;
HandleCLabel.Position = [LabelPos(1) + 1.1, LabelPos(2), LabelPos(3)];

% Plot the algorithm's path
hold on;
plot(ParamPosArray(:,2),ParamPosArray(:,1),'k','LineWidth',1.5);
% Add the start (green) and end (red) points
scatter(ParamPosArray(1,2),ParamPosArray(1,1),'g','filled');
scatter(ParamPosArray(end,2),ParamPosArray(end,1),'r','filled');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regression Error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NumMeas = length(x) % number of data points
SER = sqrt(RSS/(NumMeas - NumParams)) % SER calculation
% Compare the value to the standard deviation of the residuals
std(Residuals)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The Jacobian returned by lsqcurvefit is a sparse matrix, so convert it to a full matrix
FullJacob = full(RetJacob);
% Use a QR decomposition of the matrix
[Q,R] = qr(RetJacob,0);
% Invert the R matrix
Rinv = R \ eye(size(R));
% Get the inversion of the (J'J) regression matrix applicable to NLLS regression
JTJInverse = Rinv*Rinv';

% Multiply by the noise value (SER) earlier, but square it to get variance
CovMatrix = SER.^2 * JTJInverse;

% Get the parameter standard error values from this matrix
ParameterStdErr = sqrt(diag(CovMatrix));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Contour Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For detailed RSS plotting of parameter values
Beta1RangeDetailed = 0.3:0.005:0.7; % Set the same scale for both parameters
Beta2RangeDetailed = 4.8:0.005:5.2;
RSSForParamsDetailed = zeros(length(Beta1RangeDetailed),length(Beta2RangeDetailed));

for i = 1:length(Beta1RangeDetailed)
    for j = 1:length(Beta2RangeDetailed)
        Params = [Beta1RangeDetailed(i), Beta2RangeDetailed(j)];
        RSSForParamsDetailed(i,j) = CompareRSSOfTwoSignals(DecayCosine(Params,x),ynoise);
    end
end

% Plot a 3D contour of the RSS values
figure('Position', [50,50,700,600]);
% whitebg('k'); % if you want to set the background to black for contrast
colormap('jet'); % change the default color map for contrast
contour(Beta2RangeDetailed,Beta1RangeDetailed,squeeze(RSSForParamsDetailed(:,:)), 50); % 3D contour plot
zlim([0 20]); % Set the z dimension range to match the contour plot values
xlabel('\beta2','FontWeight','bold','fontsize', 14); % x axis label
ylabel('\beta1','FontWeight','bold','fontsize', 14); % y axis label
set(gca,'FontSize',12,'FontWeight','bold'); % Fix axis label font size
HandleC =  colorbar('location', 'EastOutside');
HandleCLabel = ylabel(HandleC,'RSS','FontSize',14,'FontWeight','bold');
% Optional code to rotate the colorbar label 180 degrees
drawnow; % MUST STICK THIS HERE TO GET THE NEXT PARTS TO WORK
LabelPos = HandleCLabel.Position; % Rotate the colorbar label
HandleCLabel.Rotation = -90; 
HandleCLabel.Position = [LabelPos(1) + 1.1, LabelPos(2), LabelPos(3)]; % Adjust the horizontal position to avoid text overlap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Also create the correlation matrix
DiagCov = 1 ./ sqrt(diag(CovMatrix));
DiagCorr = spdiags(DiagCov,0,NumParams,NumParams);
CorrMatrix = DiagCorr * CovMatrix * DiagCorr

% End of main code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fit function
function signal = DecayCosine(p, x)
% disp([p(1),p(2)]); % Uncomment for function value tracking
signal = exp(-p(1).*x).*cos(p(2).*x);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [signal,jacobian] = DecayCosineWithJac(p, x)
signal = exp(-p(1).*x).*cos(p(2).*x);
% disp([p(1),p(2)]); % uncomment this line to track the parameter values during the algorithm's descent
% Create the Jacobian by calculating the derivative of the above function with respect to the individual parameters
JacobPt1 = -x.*exp(-p(1).*x).*cos(p(2).*x); % d/dp(1) or d/da(exp(-a*x)*cos(b*x))
JacobPt2 = -x.*exp(-p(1).*x).*sin(p(2).*x); % d/dp(2) or d/db(exp(-a*x)*cos(b*x))
jacobian = [JacobPt1' JacobPt2'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function RSSReturn = CompareRSSOfTwoSignals(Signal1, Signal2)
RSSReturn = sum((Signal1 - Signal2).^2);

return;