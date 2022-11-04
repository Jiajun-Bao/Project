clear; close all; clc
% Create a VideoReader object for the first example movie file
v = VideoReader('monte_carlo_low.mp4');
v1_frames = read(v);
v1_total_frames = v.NumFrames;  % find total number of frames
% implay('monte_carlo_low.mp4')
v1 = rgb2gray(v1_frames(:,:,:,30));
imshow(v1)

%% Reshape the given data
for i = 1:v1_total_frames
    v1 = rgb2gray(v1_frames(:,:,:,i));  % convert to grey scale 
    % imshow(v1)
    v1 = reshape(v1, [], 1);
    % converted to double precision for mathematical processing
    v1_mat(:,i) = double(v1);  
end

%% DMD
X1 = v1_mat(:,1:end-1);
X2 = v1_mat(:,2:end);
[U1,Simga1,V1] = svd(X1, 'econ');

% Plot singular value spectrum
sig1 = diag(Simga1);
figure(1)
subplot(2,1,1)
plot(sig1,'o','Linewidth',1.5)
set(gca,'Fontsize',14)
title("Singular Value Spectrum (monte\_carlo\_low.mp4)","FontSize", 18);
xlabel("Singular Values"); ylabel("\sigma");
subplot(2,1,2)
plot(cumsum(sig1/sum(sig1)),'ro','Linewidth',1.5)
set(gca,'Fontsize',14, 'ylim',[0 1])
title("Cumulative Energy Captured", "FontSize", 18);
xlabel("Singular Values"); ylabel("Cumulative Energy");
hold on
plot([0 400], [0.9 0.9], '--g', 'Linewidth',1.5)
legend('Cum Energy Captured','90% Energy Captured')

%% DMD  
r = 99;
v1_total_time = v.Duration;
v1_dt = v1_total_time/v1_total_frames;
x1_t = (0:v1_total_frames-2)*v1_dt;

% Computation of ~S
U_r1 = U1(:, 1:r); % low-rank approximation (rank-r)
S_r1 = Simga1(1:r, 1:r);
V_r1 = V1(:, 1:r);
S = U_r1' * X2 * V_r1 * diag(1./diag(S_r1));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
omega = log(mu)/v1_dt;
Phi = U_r1*eV;

% Create DMD Solution
y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions
u_modes = zeros(length(y0),length(x1_t));
for iter = 1:length(x1_t)
   u_modes(:,iter) = y0.*exp(omega*x1_t(iter)); 
end
X_dmd_low = Phi*u_modes;

%% Plotting Eigenvalues (omega)
% make axis lines
line = -130:50;
plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
hold on
plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
plot(real(omega),imag(omega),'r.','Markersize',15)
xlabel('Re(\omega)')
ylabel('Im(\omega)')
title('Omega Values')
set(gca,'FontSize',16,'Xlim',[-120 10],'Ylim',[-50 50])

%%  Sparse reconstruction
X_sparse = X1 - abs(X_dmd_low);
check_neg = (X_sparse <0); % create logical index
R = X_sparse.*check_neg;
X_background = R + abs(X_dmd_low);
X_foreground = X_sparse - R;
X_reconstruct = X_background + X_foreground;

%% plot background
window_height = v.Height;
window_width = v.Width;
frame = length(x1_t);
X_dmd_low = reshape(X_dmd_low, [window_height, window_width, frame]);
X_dmd_low = uint8(X_dmd_low);
% for i = 1:frame
%     imshow(X_dmd_low(:,:,i))
% end
imshow(X_dmd_low(:,:,30))

%% plot foreground
foreground = reshape(X_foreground, [window_height, window_width, frame]);
foreground = uint8(foreground);
% for i = 1:frame
%     imshow(foreground(:,:,i))
% end
imshow(foreground(:,:,310))

%% plot reconstruction
reconstruct = reshape(X_reconstruct, [window_height, window_width, frame]);
reconstruct = uint8(reconstruct);
% for i = 1:100
%     imshow(reconstruct(:,:,i))
% end
imshow(reconstruct(:,:,310))

%% Second Vide Processing 
clear; close all; clc
% Create a VideoReader object for the second example movie file
v = VideoReader('ski_drop_low.mp4');
v1_frames = read(v);
v1_total_frames = v.NumFrames;  % find total number of frames
% implay('monte_carlo_low.mp4')
v1 = rgb2gray(v1_frames(:,:,:,430));
%imshow(v1)

%% reshape the data
for i = 1:v1_total_frames
    v1 = rgb2gray(v1_frames(:,:,:,i));  % convert to grey scale 
    % imshow(v1)
    v1 = reshape(v1, [], 1);
    % converted to double precision for mathematical processing
    v1_mat(:,i) = double(v1);  
end

%% DMD
X1 = v1_mat(:,1:end-1);
X2 = v1_mat(:,2:end);
[U1,Simga1,V1] = svd(X1, 'econ');

% Plot singular value spectrum
sig1 = diag(Simga1);
figure(1)
subplot(2,1,1)
plot(sig1,'o','Linewidth',1.5)
set(gca,'Fontsize',14)
title("Singular Value Spectrum (ski\_drop\_low.mp4)","FontSize", 18);
xlabel("Singular Values"); ylabel("\sigma");
subplot(2,1,2)
plot(cumsum(sig1/sum(sig1)),'ro','Linewidth',1.5)
set(gca,'Fontsize',14, 'ylim',[0 1])
title("Cumulative Energy Captured", "FontSize", 18);
xlabel("Singular Values"); ylabel("Cumulative Energy");
hold on
plot([0 500], [0.9 0.9], '--g', 'Linewidth',1.5)
legend('Cum Energy Captured','90% Energy Captured')

%% DMD low-rank reconstructions
r = 57;   
v1_total_time = v.Duration;
v1_dt = v1_total_time/v1_total_frames;
x1_t = (0:v1_total_frames-2)*v1_dt;

% Computation of ~S
U_r1 = U1(:, 1:r); % low-rank approximattion (rank-r)
S_r1 = Simga1(1:r, 1:r);
V_r1 = V1(:, 1:r);
S = U_r1' * X2 * V_r1 * diag(1./diag(S_r1));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
omega = log(mu)/v1_dt;
Phi = U_r1*eV;

% Create DMD Solution
y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions
u_modes = zeros(length(y0),length(x1_t));
for iter = 1:length(x1_t)
   u_modes(:,iter) = y0.*exp(omega*x1_t(iter)); 
end
X_dmd_low = Phi*u_modes;

%%  Sparse reconstruction
X_sparse = X1 - abs(X_dmd_low);
check_neg = (X_sparse <0); % create logical index
R = X_sparse.*check_neg;
X_background = R + abs(X_dmd_low);
X_foreground = X_sparse - R;
X_foreground_more_grey = X_sparse - R + 100.*ones(518400,453);
X_reconstruct = X_background + X_foreground;

%% Plotting Eigenvalues (omega)
% make axis lines
line = -80:30;
plot(zeros(length(line),1),line,'k','Linewidth',2) % imaginary axis
hold on
plot(line,zeros(length(line),1),'k','Linewidth',2) % real axis
plot(real(omega),imag(omega),'r.','Markersize',15)
xlabel('Re(\omega)')
ylabel('Im(\omega)')
title('Omega Values')
set(gca,'FontSize',16,'Xlim',[-70 5],'Ylim',[-25 25])

%% plot background
window_height = v.Height;
window_width = v.Width;
frame = length(x1_t);
X_dmd_low = reshape(X_dmd_low, [window_height, window_width, frame]);
X_dmd_low = uint8(X_dmd_low);
% for i = 1:100
%     imshow(X_dmd_low(:,:,i))
% end
imshow(X_dmd_low(:,:,430))

%% plot foreground
%plot original color scale 
%foreground = reshape(X_foreground_more_grey, [window_height, window_width, frame]);
%foreground = uint8(foreground);
%imshow(foreground(:,:,430))

% plot using different color scale uses the full range of the colormap
foreground1 = reshape(X_sparse, [window_height, window_width, frame]);
foreground1 = uint8(foreground1);
% for i = 1:150
%     imshow(foreground(:,:,i))
% end
imagesc(foreground(:,:,200))
%% plot reconstruction
reconstruct = reshape(X_reconstruct, [window_height, window_width, frame]);
reconstruct = uint8(reconstruct);
% for i = 1:100
%     imshow(reconstruct(:,:,i))
% end
imshow(reconstruct(:,:,430))
