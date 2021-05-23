%% Test Equaliser performance for PAM-2 transmission over a
%% plastic optical fibre (POF) system
%% Author: Isaac N. O. Osahon
%% Script is written with MATLAB R2020a and uses data from the following 
%% set of parameters for a 27m UWOC link:
%% 1. P_Tx = 7 dBm, G_APD = 80, Rb = 63 Mbps, M = 4
%% 2. P_Tx = 7 dBm, G_APD = 80, Rb = 125 Mbps, M = 4
%% 3. P_Tx = 10 dBm, G_APD = 100, Rb = 94 Mbps, M = 8
%% 4. P_Tx = 10 dBm, G_APD = 100, Rb = 188 Mbps, M = 8
%% 5. P_Tx = 13 dBm, G_APD = 100, Rb = 125 Mbps, M = 16
%% 6. P_Tx = 13 dBm, G_APD = 100, Rb = 250 Mbps, M = 16

clear; clc; %Clear variables and commands
%% UWOC Parameters
l_uw = 27; % Length of underwater link in metres
P_Tx = 10; % Optical transmitted power in dBm
G_APD = 100; % Avalanche Photodiode Gain
Rb = 188; % Bit rate in Mbps
%% PAM Parameters
M = 8; % PAM Constellation size
k = log2(M);
Cnstln = 1-M:2:M-1; % Symbol Constellation

%% File Parameters
% The current folder should be "C:\.............\UWOCData for Windows
dirStr = ''; %Directory string
fileStr = ['UWOC',int2str(l_uw),'m_PAM',int2str(M),'_',int2str(Rb),...
    'Mb_APDGain',int2str(G_APD),'_P',int2str(P_Tx),...
    'dBm.mat']; % String for file name

% Load file variables and covert to double
%(PAMsymTx = Transmitted PAM symbols, PAMsymRx = Received PAM symbols)
load([dirStr,fileStr],'PAMsymTx','PAMsymRx');
PAMSymTx = double(PAMsymTx);
PAMSymRx = double(PAMsymRx);

%eyediagram(PAMSymTx,M);
% figure(1);
% plot(PAMSymTx), title('Raw Tx');
% xlim([0,50])

% figure(2);
% plot(PAMSymRx), title('Raw Rx');
% xlim([0,50]);
%disp(PAMSymTx);
nSym = length(PAMSymTx); % Number of PAM symbols

%a = pamdemod(PAMSymTx,M,0,'gray');
%disp(a);
%% Evaluate BER without equalizer
nTr1 = 500;
[~,~,WghtVec,~] = myRLSEQ(1,0,0,PAMSymRx(1:nTr1),PAMSymTx(1:nTr1),...
    Cnstln,1,1,'Ndd');
PAMSymRxNm1 = WghtVec(1)+ WghtVec(2)*PAMSymRx;

%eyediagram(PAMSymRxNm1,M);

% figure(3);
% plot(PAMSymRxNm1), title('weight vector Rx');
% xlim([0,50]);

deModulatedTx = pamdemod(PAMSymTx,M,0,'gray');
% figure(4);
% plot(deModulatedTx), title('de mod TX');
% xlim([0,50]);

TxBit =reshape((de2bi(deModulatedTx,k,[],'left-msb'))',1,k*nSym);
% figure(5);
% plot(TxBit), title('Tx to binary');
% xlim([0,50]);
deModulatedRx = pamdemod(PAMSymRxNm1,M,0,'gray')
% figure(6);
% plot(deModulatedTx), title('de mod RX');
% xlim([0,50]);
RxBit0 = reshape((de2bi(pamdemod(PAMSymRxNm1,M,0,'gray'),k,[],...
    'left-msb'))',1,k*nSym);

%figure(2);
%plot(RxBit0), title('Reshape output');
%xlim([0,50]);


[~,BERNoEq] = symerr(TxBit,RxBit0);
fprintf('The BER of the system without equalisation is %e\n\n', BERNoEq)

%% Evaluate BER with the conventional decision feedback equalizer (DFE)
nft = 16; % Number of forward taps
eqDel = round(nft/2); % Equalizer tap delay
nbt = 8;  % Number of feedback taps
nTrEQ = 4e3; % Number of training symbols
PAMSymRxEQ1 = myRLSEQ(nft,nbt,eqDel,PAMSymRx,PAMSymTx(1:nTr1),...
    Cnstln,1,1,'Ndd');
nSymTest = nSym-nTrEQ-eqDel;




% figure(6);
% plot(PAMSymRxEQ1), title('After Equalisation');
% xlim([0,50])

TxBit = TxBit(k*nTrEQ+(1:k*nSymTest));

a = pamdemod(PAMSymRxEQ1(nTrEQ+eqDel+1:end),M,0,'gray');
% figure(7);
% plot(a), title('de modulated Rx');
% xlim([0,50])



RxBit1 = reshape((de2bi(pamdemod(PAMSymRxEQ1(nTrEQ+eqDel+1:end),...
    M,0,'gray'),k,[],'left-msb'))',1,k*nSymTest);


[~,BEREq] = symerr(TxBit,RxBit1);
fprintf('The BER of the system with the coventional DFE is %e\n\n', BEREq)



