% In this example, we design and implement a length L = 257 FIR lowpass
% filter having a cut-off frequency at f_c = 300 Hz. The filter is tested
% on an input signal x(n) consisting of a sum of sinusoidal components at
% frequencies  (220, 250, 600, 800) Hz. We'll filter a single input
% frame of length M = 256 ,
% which allows the FFT to be N=512 samples (no wasted zero-padding).

% Signal parameters:
f = [ 220 250 600 800];      % frequencies   
M = 256;                        % signal length 
Fs = 2000;                      % sampling rate

    
% Generate a signal by adding up sinusoids:
x = zeros(1,M);        % pre-allocate
x_target = zeros(1,M); % pre-allocate
n = 0:(M-1);           % discrete-time grid 
for fk = f
    x = x + sin(2*pi*n*fk/Fs);
    if fk < 300
        x_target = x_target + sin(2*pi*n*fk/Fs);
    end
end

%% Let's see how our original and target signal look like:
figure();
subplot(2,1,1);
plot(n,x);
title("original signal");
subplot(2,1,2);
plot(n,x_target);
title("target signal");

%%

% Next we design the lowpass filter using the window method:
% The window method consists of simply "windowing" a theoretically ideal 
% filter impulse response hideal(n) by some suitably chosen window function
% w(n) ()hamming window in the example below)
% Filter parameters:
L = 257;    % filter length 
fc = 300;   % cutoff frequency 

% Design the filter using the window method:
hsupp = (-(L-1)/2:(L-1)/2);
% This way we have that 
% hsupp = -128,127,126,...,126,127,128
hideal = (2*fc/Fs)*sinc(2*fc*hsupp/Fs);
h = hamming(L)' .* hideal; % h is our filter

%% Visualize our filter in both time and frequency domains
% This toolbox uses the convention that unit frequency is the Nyquist
% frequency, defined as half the sampling frequency.
% The cutoff frequency parameter for all basic filter design functions
% is normalized by the Nyquist frequency. For a system with a 1000 Hz
% sampling frequency, for example, 300 Hz is 300/500 = 0.6.
% In our case we have a sampling frequency equal to 2000, so our Nyquist is
% 1000 Hz, i.e. the cutoff frequency Fc = 300 Hz is equal to 0.3.
wvtool(h);

%% Filter in the time domain

x_filtered = conv(x, h ,'same');

figure();
subplot(3,1,1);
plot(n,x);
title("original signal");
subplot(3,1,2);
plot(n,x_filtered);
title("filtered signal");
subplot(3,1,3);
plot(n,x_target);
title("target signal");

%% Filter in frequency time

% Choose the next power of 2 greater than L+M-1 
Nfft = 2^(ceil(log2(L+M-1)));

% Zero pad the signal and impulse response:
xzp = [ x zeros(1,Nfft-M)];
hzp = [ h zeros(1,Nfft-L)];

X = fft(xzp); % signal transform
H = fft(hzp); % filter transform
figure();

frequencies= linspace(-Fs/2 , +Fs/2 , 512);
subplot(2,1,1)
plot(frequencies,abs(fftshift(X)));
title("Transform of original signal")
subplot(2,1,2)
plot(frequencies,abs(fftshift(H)));
title("filter in Frequency domain")


%%
% Now we perform cyclic convolution in the time domain using pointwise
% multiplication in the frequency domain:

X_filtered2 = X .* H;
x_filtered2 = ifft(X_filtered2);
x_filtered2 = real(x_filtered2);

% TO DO: APPARENTLY THERE IS A SHIFT IN TIME OF 128 BINS IN THE FILTERED
% SIGNAL, FIND OUT WHY!
figure();
subplot(3,1,1);
plot(n,x);
title("original signal");
subplot(3,1,2);
plot(-128:511-128,x_filtered2);
title("filtered signal");
subplot(3,1,3);
plot(n,x_target);
title("target signal");
