
M = 8;
k = log2(M);
Cnstln = 1-M:2:M-1; % Symbol Constellation
dirStr = '';
fileStr = 'equalizedOutput.mat';
load([dirStr,fileStr],'Tx','Rx','input');
Tx = double(Tx);
Rx = double(Rx);
input = double(input);
nSym = length(Tx);
%% eye diagram
sps = 1;
span = 10;
%eyediagram(Tx(sps*span+1:sps*span + 750),2), title('');
%eyediagram(Rx(sps*span+1:sps*span + 750),2), title('LSTM Equalized');
%eyediagram(input(sps*span+1:sps*span + 1500),2), title('2PAM@600Mbps Raw Rx');
%% demodulate TX
deModulatedTx = pamdemod(Tx,M,0,'gray');
TxBit =reshape((de2bi(deModulatedTx,k,[],'left-msb'))',1,k*nSym);

%% No Equalization
nTr1 = 500;

[~,~,WghtVec,~] = myRLSEQ(1,0,0,input(1:nTr1),Tx(1:nTr1),...
    Cnstln,1,1,'Ndd');
PAMSymRxNm1 = WghtVec(1)+ WghtVec(2)*input;


%eyediagram(PAMSymRxNm1(sps*span+1:sps*span + 1500),2), title('Recived Signal');
deModulatedInput = pamdemod(PAMSymRxNm1,M,0,'gray');

inputBit = reshape((de2bi(deModulatedInput,k,[],'left-msb'))',1,k*nSym);
[~, BEReq] = symerr(TxBit, inputBit);
fprintf('The BER of the system with no equalisation is %e\n\n', BEReq)

%% DFE Equalizer
tic
 nft = 16; % Number of forward taps
 eqDel = round(nft/2); % Equalizer tap delay
nbt = 8;  % Number of feedback taps
 nTrEQ = 4e3; % Number of training symbols
PAMSymRxEQ1 = myRLSEQ(nft,nbt,eqDel,input,Tx(1:nTr1),... %goes through the equalizer
   Cnstln,1,1,'Ndd');
 nSymTest = nSym-nTrEQ-eqDel;


%eyediagram(PAMSymRxEQ1(sps*span+1:sps*span + 750),2), title('DFE Equalized Rx');

 TxBit1 = TxBit(k*nTrEQ+(1:k*nSymTest)); %reduce size of Tx so same size as DFE output

RxBit1 = reshape((de2bi(pamdemod(PAMSymRxEQ1(nTrEQ+eqDel+1:end),...
    M,0,'gray'),k,[],'left-msb'))',1,k*nSymTest);
[~,BEREq] = symerr(TxBit1,RxBit1);

fprintf('The BER of the system with the coventional DFE is %e\n\n', BEREq);
toc
%% ML Equalization
[~,~,WghtVec1,~] = myRLSEQ(1,0,0,Rx(1:nTr1),Tx(1:nTr1),...
    Cnstln,1,1,'Ndd');
%PAMSymRxNm2 = WghtVec1(1)+ WghtVec1(2)*Rx;

deModulatedRx = pamdemod(Rx,M,0,'gray');
RxBit =reshape((de2bi(deModulatedRx,k,[],'left-msb'))',1,k*nSym);

RxBit2 = RxBit(k*nTrEQ+(1:k*nSymTest));

[~, BEReq2] = symerr(TxBit1, RxBit2);

fprintf('The BER of the system with ML equalisation is %e\n\n', BEReq2)

%% MSE Eqn

function returns = MSE(X,Y)
returns = mean((X-Y).^2);
end

