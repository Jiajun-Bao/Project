% Clean workspace
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata 5
L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    M = max(abs(Un),[],'all');
    close all, isosurface(X,Y,Z,abs(Un)/M,0.7) 
    axis([-20 20 -20 20 -20 20]), grid on, drawnow 
    pause(1)
end

% plot raw signals
for j=1:49
    un_raw(:,:,:)=reshape(subdata(:,j),n,n,n);
    unf_raw = fftn(un_raw);
end
Max = max(abs(unf_raw),[],'all');
figure 
isosurface(Kx,Ky,Kz, abs(unf_raw./Max), 0.4)   
axis([-10 10 -10 10 -10 10]), grid on, drawnow 
xlabel('kx')
ylabel('ky')
zlabel('kz')
title ('Normalized Raw Signals')
set(gca,'FontSize',14)
print(gcf,'-dpng','fig1.png')

% average over realizations in frequency spaces 
u_ave = zeros(n,n,n);
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    unf = fftn(Un);
    u_ave = u_ave + unf;
end
u_ave = abs(fftshift(u_ave))./49;

% find the frequency signature (center frequency)
[M,I] = max(u_ave(:));
[I1,I2,I3] = ind2sub(size(u_ave),I);

% plot averaged signals
figure 
isosurface(Kx,Ky,Kz, abs(u_ave)./M, 0.5)   
axis([-10 10 -10 10 -10 10]), grid on, drawnow 
xlabel('kx')
ylabel('ky')
zlabel('kz')
title ('Normalized Averaged Signals')
set(gca,'FontSize',14)
print(gcf,'-dpng','fig2.png')

%location of peak frequency
peak_x = Kx(I1,I2,I3);
peak_y = Ky(I1,I2,I3);
peak_z = Kz(I1,I2,I3);

% Define the filter 
tau = 0.5;
filter = exp(-tau*((Kx-peak_x).^2 + (Ky-peak_y).^2 + (Kz-peak_z).^2));
X_path = zeros(49,1);
Y_path = zeros(49,1);
Z_path = zeros(49,1);

for j = 1:49
    un = reshape(subdata(:,j),n,n,n); % Noisy signal in time space
    utn = fftn(un); % Noisy signal in frequency space
    utn = fftshift(utn);
    unft = filter .* utn; % Apply the filter to the signal in frequency space 
    unf = ifftn(unft); % Inverse back to time space
    [M1,Ind] = max(unf(:));  
    [X_x,Y_y,Z_z] = ind2sub(size(unf),Ind);
    X_path(j,1) = X(X_x, Y_y, Z_z);
    Y_path(j,1) = Y(X_x, Y_y, Z_z);
    Z_path(j,1) = Z(X_x, Y_y, Z_z);
end

% plot the path of the submarine in time space
figure 
plot3(X_path, Y_path, Z_path,'-o','Color','b','LineWidth', 1.5)
title('Path of the Submarine in Spacial Space','FontSize', 30)
xlabel('X','FontSize', 12)
ylabel('Y','FontSize', 12)
zlabel('Z','FontSize', 12)
set(gca,'FontSize',12)
print(gcf,'-dpng','fig3.png')