%% Part 1.1(a) Music score for the guitar in the GNR clip
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Sweet Child O'' Mine');
p8 = audioplayer(y,Fs); playblocking(p8);
L = tr_gnr; n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);
t2 = linspace(0,L,n+1); t = t2(1:n);

% Produce the GNR Spectrogram
a = 50;
tau = 0:0.1:tr_gnr;
freq_k= [];

for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   Sg = g.*y';
   Sgt = fft(Sg);
   Sgt_spec(:,j) = fftshift(abs(Sgt)); 
   [M,I] = max(abs(Sgt));
   freq_k(j) = abs(k(I));
end

figure(2)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0 1000],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('GNR Spectrogram')
yyaxis right
ylabel('Music score for the guitar')
yticks([277.0, 311.0, 369.0, 415.0, 556.0, 701.0, 742.0])
yticklabels({'C#4','D#4','F#4','G#4','C#5','F5','F#5'})
print(gcf,'-dpng','GNR Spectrogram.png')

% Produce the music score diagram for the guitar in the GNR clip
figure(3)
plot(tau, freq_k, '.','Color','b','LineWidth', 3)
title('Music score for the guitar in the GNR clip','FontSize', 30)
ylim([150 900])
xlabel('time (sec)','FontSize', 12)
ylabel('frequency (Hz)','FontSize', 12)
yyaxis right
ylabel('Music score for the guitar')
yticks([277.0, 311.0, 369.0, 415.0, 556.0, 701.0, 742.0])
yticklabels({'C#4','D#4','F#4','G#4','C#5','F5','F#5'})
set(gca,'FontSize',12)
print(gcf,'-dpng','fig3.png')


%% Part 1.1(b)  Music score for the guitar in the GNR clip w/o overtone
close all; clear all; clc;
figure(1)
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
L = tr_gnr; n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);
t2 = linspace(0,L,n+1); t = t2(1:n);
a = 50;
tau = 0:0.1:tr_gnr;

for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2); % Window function
   Sg = g.*y';
   Sgt = fft(Sg);
   Sgt_spec(:,j) = fftshift(abs(Sgt); 
   [M1,I1] = max(abs(Sgt));
   filter = exp(-0.01*(k - k(I1)).^2);
   Sgtf = filter.*Sgt;
   Sgt_spec(:,j) = fftshift(abs(Sgtf)); 
end

figure(1)
pcolor(tau,ks,Sgt_spec)
shading interp
set(gca,'ylim',[0 1000],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('GNR Spectrogram w/o Overtones')
yyaxis right
ylabel('Music score for the guitar')
yticks([277.0, 311.0, 369.0, 415.0, 556.0, 701.0, 742.0])
yticklabels({'C#4','D#4','F#4','G#4','C#5','F5','F#5'})
print(gcf,'-dpng','GNR Spectrogram no overtone.png')

%% Part 1.2 Music score for the bass in the Floyd clip
close all; clear all; clc;
figure(1)
[y, Fs] = audioread('Floyd.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Comfortably Numb');
p8 = audioplayer(y,Fs); playblocking(p8);
L = tr_gnr; n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);
t2 = linspace(0,L,n+1); t = t2(1:n-1);

% Divide Floyd Spectrogram into 4 time periods with 25s each
k1 = (1/15)*[0:n/8-1 -n/8:-1];
ks1 = fftshift(k1);
k2 = (1/15)*[0:n/8-1 -n/8:-1];
ks2 = fftshift(k2);
k3 = (1/15)*[0:n/8-1 -n/8:-1];
ks3 = fftshift(k3);
k4 = (1/(L-45))*[0:n/8-1 -n/8:-1];
ks4 = fftshift(k4);
y1 = y(1:(length(y)-1)/4);
y2 = y((length(y)-1)/4+1:(length(y)-1)/2);
y3 = y((length(y)-1)/2+1:3*(length(y)-1)/4);
y4 = y(3*(length(y)-1)/4+1:length(y)-1);
t_1st_quarter = t(1:length(t)/4);
t_2nd_quarter = t(length(t)/4+1: length(t)/2);
t_3rd_quarter = t(length(t)/2+1:3*length(t)/4);
t_4th_quarter = t(3*length(t)/4+1:length(t));
tau1 = 0:0.1:15;
tau2 = 15:0.1:30;
tau3 = 30:0.1:45;
tau4 = 45:0.1:tr_gnr;

a = 100;
% Floyd 0 - 15s Spectrogram 
for j = 1:length(tau1)
   g1 = exp(-a*(t_1st_quarter - tau1(j)).^2); % Window function
   Sg1 = g1.*y1';
   Sgt1 = fft(Sg1);
   Sgt_spec1(:,j) = fftshift(abs(Sgt1)); 
end

figure(2)
pcolor(tau1,ks1,Sgt_spec1)
shading interp
set(gca,'ylim',[0 300],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram 1-15s')
print(gcf,'-dpng',' Floyd Spectrogram 1-15.png')

% Floyd 15 - 30s Spectrogram 
for j = 1:length(tau2)
   g2 = exp(-a*(t_2nd_quarter - tau2(j)).^2); % Window function
   Sg2 = g2.*y2';
   Sgt2 = fft(Sg2);
   Sgt_spec2(:,j) = fftshift(abs(Sgt2)); 
end

figure(3)
pcolor(tau2,ks2,Sgt_spec2)
shading interp
set(gca,'ylim',[0 300],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram 15-30s')
print(gcf,'-dpng',' Floyd Spectrogram 15-30s.png')

% Floyd 30 - 45s Spectrogram 
for j = 1:length(tau3)
   g3 = exp(-a*(t_3rd_quarter - tau3(j)).^2); % Window function
   Sg3 = g3.*y3';
   Sgt3 = fft(Sg3);
   Sgt_spec3(:,j) = fftshift(abs(Sgt3)); 
end
figure(4)
pcolor(tau3,ks3,Sgt_spec3)
shading interp
set(gca,'ylim',[0 300],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram 30-45s')
print(gcf,'-dpng',' Floyd Spectrogram 30-45s.png')

% Floyd 45 - 60s Spectrogram 
for j = 1:length(tau4)
   g4 = exp(-a*(t_4th_quarter - tau4(j)).^2); % Window function
   Sg4 = g4.*y4';
   Sgt4 = fft(Sg4);
   Sgt_spec4(:,j) = fftshift(abs(Sgt4)); 
end
figure(5)
pcolor(tau4,ks4,Sgt_spec4)
shading interp
set(gca,'ylim',[0 300],'Fontsize',16)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram 45-60s')
print(gcf,'-dpng',' Floyd Spectrogram 45-60s.png')

%% Part 2 Music score for the bass in the Floyd clip w/o overtone
close all; clear all; clc;
figure(1)
[y, Fs] = audioread('Floyd.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title('Comfortably Numb');
p8 = audioplayer(y,Fs); playblocking(p8);
L = tr_gnr; n = length(y);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);
t2 = linspace(0,L,n+1); t = t2(1:n-1);

% Divide Floyd Clip into four 15-secs periods 
k1 = (1/15)*[0:n/8-1 -n/8:-1];
ks1 = fftshift(k1);
k2 = (1/15)*[0:n/8-1 -n/8:-1];
ks2 = fftshift(k2);
k3 = (1/15)*[0:n/8-1 -n/8:-1];
ks3 = fftshift(k3);
k4 = (1/(L-45))*[0:n/8-1 -n/8:-1];
ks4 = fftshift(k4);
y1 = y(1:(length(y)-1)/4);
y2 = y((length(y)-1)/4+1:(length(y)-1)/2);
y3 = y((length(y)-1)/2+1:3*(length(y)-1)/4);
y4 = y(3*(length(y)-1)/4+1:length(y)-1);
t_1st_quarter = t(1:length(t)/4);
t_2nd_quarter = t(length(t)/4+1: length(t)/2);
t_3rd_quarter = t(length(t)/2+1:3*length(t)/4);
t_4th_quarter = t(3*length(t)/4+1:length(t));
tau1 = 0:0.1:15;
tau2 = 15:0.1:30;
tau3 = 30:0.1:45;
tau4 = 45:0.1:tr_gnr;
a = 50;
% Floyd 1 - 15s w/o overtone Spectrogram + Score
freq_k1= [];
for j = 1:length(tau1)
   g1 = exp(-a*(t_1st_quarter - tau1(j)).^2); % Window function
   Sg1 = g1.*y1';
   Sgt1 = fft(Sg1);
   [M1,I1] = max(abs(Sgt1));
   filter1 = exp(-0.01*(k1 - k1(I1)).^2);
   Sgtf1 = filter1.*Sgt1;
   [M,I] = max(abs(Sgtf1));
   freq_k1(j) = abs(k1(I));
   Sgt_spec1(:,j) = fftshift(abs(Sgtf1));  
end
figure(2)
pcolor(tau1,ks1,Sgt_spec1)
shading interp
set(gca,'ylim',[0 200],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram w/o Overtones 1-15s')
yyaxis right
ylabel('Music score for bass')
yticks([82.0, 90.7, 97.3, 110.0, 124.0])
yticklabels({'E2','F#2','G2','A2','B2'})
print(gcf,'-dpng','1-15_no_overtone.png')
figure(3)
plot(tau1, freq_k1, '.','Color','b','LineWidth', 3)
title('Music score for Bass in Floyd 1-15s Clip','FontSize',12)
ylim([0 200])
xlabel('time (sec)','FontSize', 12)
ylabel('frequency (Hz)','FontSize', 12)
yyaxis right
ylabel('Music score for bass')
yticks([82.0, 90.7, 97.3, 110.0, 124.0])
yticklabels({'E2','F#2','G2','A2','B2'})
set(gca,'FontSize',12)
print(gcf,'-dpng','1-15_notes.png')

% Floyd 15 - 30s w/o overtone Spectrogram + Score
freq_k2= [];
for j = 1:length(tau2)
   g2 = exp(-a*(t_2nd_quarter - tau2(j)).^2); % Window function
   Sg2 = g2.*y2';
   Sgt2 = fft(Sg2);
   [M2,I2] = max(abs(Sgt2));
   filter2 = exp(-0.01*(k2 - k2(I2)).^2);
   Sgtf2 = filter2.*Sgt2;
   [M,I] = max(abs(Sgtf2));
   freq_k2(j) = abs(k2(I));
   Sgt_spec2(:,j) = fftshift(abs(Sgtf2)); 
end
figure(4)
pcolor(tau2,ks2,Sgt_spec2)
shading interp
set(gca,'ylim',[0 200],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram w/o Overtones 15-30s')
yyaxis right
ylabel('Music score for bass')
yticks([82.0, 90.7, 97.3, 110.0, 124.0])
yticklabels({'E2','F#2','G2','A2','B2'})
print(gcf,'-dpng','15-30_no_overtone.png.png')
figure(5)
plot(tau2, freq_k2, '.','Color','b','LineWidth', 3)
title('Music score for Bass in Floyd 15-30s Clip','FontSize', 12)
ylim([0 200])
xlabel('time (sec)','FontSize', 12)
ylabel('frequency (Hz)','FontSize', 12)
yyaxis right
ylabel('Music score for bass')
yticks(  )
yticklabels({'E2','F#2','G2','A2','B2'})
set(gca,'FontSize',12)
print(gcf,'-dpng','15-30_notes.png')

% Floyd 30 - 45s w/o overtone Spectrogram + Score
for j = 1:length(tau3)
   g3 = exp(-a*(t_3rd_quarter - tau3(j)).^2); % Window function
   Sg3 = g3.*y3';
   Sgt3 = fft(Sg3);
   [M3,I3] = max(abs(Sgt3));
   filter3 = exp(-0.01*(k4 - k3(I3)).^2);
   Sgtf3 = filter3.*Sgt3;
   Sgt_spec3(:,j) = fftshift(abs(Sgtf3)); 
end
figure(6)
pcolor(tau3,ks3,Sgt_spec3)
shading interp
set(gca,'ylim',[0 200],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram w/o Overtones 30-45s')
yyaxis right
ylabel('Music score for bass')
yticks([82.0, 90.7, 97.3, 110.0, 124.0])
yticklabels({'E2','F#2','G2','A2','B2'})
print(gcf,'-dpng','30-45_no_overtone.png')

% Floyd 45 - 60s w/o overtone Spectrogram + Score
figure(7)
for j = 1:length(tau4)
   g4 = exp(-a*(t_4th_quarter - tau4(j)).^2); % Window function
   Sg4 = g4.*y4';
   Sgt4 = fft(Sg4);
   [M4,I4] = max(abs(Sgt4));
   filter4 = exp(-0.01*(k4 - k4(I4)).^2);
   Sgtf4 = filter4.*Sgt4;
   Sgt_spec4(:,j) = fftshift(abs(Sgtf4)); 
end
pcolor(tau4,ks4,Sgt_spec4)
shading interp
set(gca,'ylim',[0 200],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram w/o Overtones 45-60s')
yyaxis right
ylabel('Music score for bass')
yticks([82.0, 90.7, 97.3, 110.0, 124.0])
yticklabels({'E2','F#2','G2','A2','B2'})
print(gcf,'-dpng','45-60_no_overtone.png')
%% Find Guitar solo
% Floyd 0 - 15s Spectrogram for Guitar  
freq_k1= [];
for j = 1:length(tau1)
   g1 = exp(-a*(t_1st_quarter - tau1(j)).^2); % Window function
   Sg1 = g1.*y1';
   Sgt1 = fft(Sg1);
   [M1,I1] = max(abs(Sgt1));
   filter1 = exp(-0.01*(k1 - k1(I1)).^2);
   Sgtf1 = filter1.*Sgt1;
   [M,I] = max(abs(Sgtf1));
   freq_k1(j) = abs(k1(I));
   Sgt_spec1(:,j) = fftshift(abs(Sgtf1));  
end
figure(2)
pcolor(tau1,ks1,Sgt_spec1)
shading interp
set(gca,'ylim',[150 1000],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram 1-15s for Guitar')
yyaxis right
ylabel('Music score for guitar')
yticks([246.2, 330.0, 372.0, 589.0])
yticklabels({'B3','E4','F#4','D5'})
print(gcf,'-dpng','1-15_guitar.png')
err_bond = 3;
target = [82.0, 90.7, 97.3, 110.0, 124.0];
for i = 1:length(freq_k1)
    for j = 1:length(target)
        if abs(freq_k1(i)-target(j)) < err_bond
            freq_k1(i) = 0;   
        end
    end
end
figure(3)
plot(tau1, freq_k1, '.','Color','b','LineWidth', 3)
title('Music score for Guitar in Floyd 1-15s Clip','FontSize', 12)
ylim([0 1000])
xlabel('time (sec)','FontSize', 12)
ylabel('frequency (Hz)','FontSize', 12)
yyaxis right
ylabel('Music score for guitar')
yticks([246.2, 330.0, 372.0, 589.0])
yticklabels({'B3','E4','F#4','D5'})
print(gcf,'-dpng','1-15_guitar_notes.png')

% Floyd 15 - 30s Spectrogram 
freq_k2= [];
for j = 1:length(tau2)
   g2 = exp(-a*(t_2nd_quarter - tau2(j)).^2); % Window function
   Sg2 = g2.*y2';
   Sgt2 = fft(Sg2);
   [M2,I2] = max(abs(Sgt2));
   filter2 = exp(-0.01*(k2 - k2(I2)).^2);
   Sgtf2 = filter2.*Sgt2;
   [M,I] = max(abs(Sgtf2));
   freq_k2(j) = abs(k2(I));
   Sgt_spec2(:,j) = fftshift(abs(Sgtf2));  
end
figure(4)
pcolor(tau2,ks2,Sgt_spec2)
shading interp
set(gca,'ylim',[150 1000],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram 15-30s for Guitar')
yyaxis right
ylabel('Music score for guitar')
yticks([164.5, 246.2, 372.0, 589.0, 749.0])
yticklabels({'E3','B3','F#4','D5','F5'})
print(gcf,'-dpng','15-30_guitar.png')
err_bond = 3;
target = [82.0, 90.7, 97.3, 110.0, 124.0];
for i = 1:length(freq_k2)
    for j = 1:length(target)
        if abs(freq_k2(i)-target(j)) < err_bond
            freq_k2(i) = 0;   
        end
    end
end
figure(5)
plot(tau2, freq_k2, '.','Color','b','LineWidth', 3)
title('Music score for Guitar in Floyd 15-30s Clip','FontSize', 12)
ylim([0 1000])
xlabel('time (sec)','FontSize', 12)
ylabel('frequency (Hz)','FontSize', 12)
yyaxis right
ylabel('Music score for guitar')
yticks([164.5, 246.2, 372.0, 589.0, 749.0])
yticklabels({'E3','B3','F#4','D5','F5'})
print(gcf,'-dpng','15-30_guitar_notes.png')

%Floyd 30 - 45s Spectrogram 
for j = 1:length(tau3)
   g3 = exp(-a*(t_3rd_quarter - tau3(j)).^2); % Window function
   Sg3 = g3.*y3';
   Sgt3 = fft(Sg3);
   [M3,I3] = max(abs(Sgt3));
   filter3 = exp(-0.01*(k4 - k3(I3)).^2);
   Sgtf3 = filter3.*Sgt3;
   Sgt_spec3(:,j) = fftshift(abs(Sgtf3)); 
end
figure(6)
pcolor(tau3,ks3,Sgt_spec3)
shading interp
set(gca,'ylim',[150 1000],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram 30-45s for Guitar')
yyaxis right
ylabel('Music score for guitar')
yticks([164.5, 246.2, 330.0, 372.0, 589.0])
yticklabels({'E3','B3','E4','F#4','D5'})
print(gcf,'-dpng','30-45_guitar.png')

% Floyd 45 - 60s Spectrogram 
for j = 1:length(tau4)
   g4 = exp(-a*(t_4th_quarter - tau4(j)).^2); % Window function
   Sg4 = g4.*y4';
   Sgt4 = fft(Sg4);
   [M4,I4] = max(abs(Sgt4));
   filter4 = exp(-0.01*(k4 - k4(I4)).^2);
   Sgtf4 = filter4.*Sgt4;
   Sgt_spec4(:,j) = fftshift(abs(Sgtf4)); 
end
figure(7)
pcolor(tau4,ks4,Sgt_spec4)
shading interp
set(gca,'ylim',[150 1000],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (sec)'), ylabel('frequency (Hz)')
title ('Floyd Spectrogram 45-60s for Guitar')
yyaxis right
ylabel('Music score for guitar')
yticks([164.5, 246.2, 330.0, 372.0, 589.0])
yticklabels({'E3','B3','E4','F#4','D5'})
print(gcf,'-dpng','45-60_guitar.png')