Ľ
ĚŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-0-gb36436b0878Ś

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
}
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*(
_output_shapes
:*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
É	*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
É	*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*'
_output_shapes
:*
dtype0
}
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
v
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
z
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
z
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes	
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
É	*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
É	*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*'
_output_shapes
:*
dtype0
}
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
v
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
z
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
z
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes	
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
É	*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
É	*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
žE
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ůD
valueďDBěD BĺD

layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
layer-13
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
 trainable_variables
!	variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
R
)regularization_losses
*trainable_variables
+	variables
,	keras_api
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
R
1regularization_losses
2trainable_variables
3	variables
4	keras_api
h

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
R
;regularization_losses
<trainable_variables
=	variables
>	keras_api
R
?regularization_losses
@trainable_variables
A	variables
B	keras_api
R
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
R
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
h

Kkernel
Lbias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
R
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
ŕ
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratem°mą#m˛$mł5m´6mľKmśLmˇv¸vš#vş$vť5vź6v˝KvžLvż
 
8
0
1
#2
$3
54
65
K6
L7
8
0
1
#2
$3
54
65
K6
L7
­
regularization_losses
trainable_variables
	variables
Zmetrics
[layer_metrics

\layers
]non_trainable_variables
^layer_regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
trainable_variables
	variables
_metrics
`layer_metrics

alayers
bnon_trainable_variables
clayer_regularization_losses
 
 
 
­
regularization_losses
trainable_variables
	variables
dmetrics
elayer_metrics

flayers
gnon_trainable_variables
hlayer_regularization_losses
 
 
 
­
regularization_losses
 trainable_variables
!	variables
imetrics
jlayer_metrics

klayers
lnon_trainable_variables
mlayer_regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
­
%regularization_losses
&trainable_variables
'	variables
nmetrics
olayer_metrics

players
qnon_trainable_variables
rlayer_regularization_losses
 
 
 
­
)regularization_losses
*trainable_variables
+	variables
smetrics
tlayer_metrics

ulayers
vnon_trainable_variables
wlayer_regularization_losses
 
 
 
­
-regularization_losses
.trainable_variables
/	variables
xmetrics
ylayer_metrics

zlayers
{non_trainable_variables
|layer_regularization_losses
 
 
 
Ż
1regularization_losses
2trainable_variables
3	variables
}metrics
~layer_metrics

layers
non_trainable_variables
 layer_regularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
˛
7regularization_losses
8trainable_variables
9	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 
 
˛
;regularization_losses
<trainable_variables
=	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 
 
˛
?regularization_losses
@trainable_variables
A	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 
 
˛
Cregularization_losses
Dtrainable_variables
E	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 
 
˛
Gregularization_losses
Htrainable_variables
I	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1

K0
L1
˛
Mregularization_losses
Ntrainable_variables
O	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
 
 
 
˛
Qregularization_losses
Rtrainable_variables
S	variables
 metrics
Ąlayer_metrics
˘layers
Łnon_trainable_variables
 ¤layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

Ľ0
Ś1
 
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

§total

¨count
Š	variables
Ş	keras_api
I

Ťtotal

Źcount
­
_fn_kwargs
Ž	variables
Ż	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

§0
¨1

Š	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ť0
Ź1

Ž	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv2d_inputPlaceholder*1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ*
dtype0*&
shape:˙˙˙˙˙˙˙˙˙ŹŹ
Ŕ
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_918963
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_919400
ń
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/dense/kernel/vAdam/dense/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_919509
Ú
d
H__inference_activation_2_layer_call_and_return_conditional_losses_919233

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙GG:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG
 
_user_specified_nameinputs
ŕ
b
F__inference_activation_layer_call_and_return_conditional_losses_918584

inputs
identityY
ReluReluinputs*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2

Identity"
identityIdentity:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ
 
_user_specified_nameinputs
¤5

F__inference_sequential_layer_call_and_return_conditional_losses_918913

inputs
conv2d_918882
conv2d_918884
conv2d_1_918889
conv2d_1_918891
conv2d_2_918897
conv2d_2_918899
dense_918906
dense_918908
identity˘conv2d/StatefulPartitionedCall˘ conv2d_1/StatefulPartitionedCall˘ conv2d_2/StatefulPartitionedCall˘dense/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_918882conv2d_918884*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_9185632 
conv2d/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_9185842
activation/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_9185192
max_pooling2d/PartitionedCallÂ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_918889conv2d_1_918891*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_9186032"
 conv2d_1/StatefulPartitionedCall
dropout/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_9186362
dropout/PartitionedCall
activation_1/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_9186542
activation_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙II* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9185312!
max_pooling2d_1/PartitionedCallÂ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_918897conv2d_2_918899*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_9186732"
 conv2d_2/StatefulPartitionedCall
dropout_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_9187062
dropout_1/PartitionedCall
activation_2/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_9187242
activation_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙##* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9185432!
max_pooling2d_2/PartitionedCallř
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_9187392
flatten/PartitionedCall˘
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_918906dense_918908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9187572
dense/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_9187782
activation_3/PartitionedCall
IdentityIdentity%activation_3/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
 
_user_specified_nameinputs
¸
Ú
+__inference_sequential_layer_call_fn_919097

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity˘StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9189132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
 
_user_specified_nameinputs
ě
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_918706

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙GG:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG
 
_user_specified_nameinputs
Ž
Ź
D__inference_conv2d_2_layer_call_and_return_conditional_losses_919192

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpĽ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙II:::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙II
 
_user_specified_nameinputs
Ů1

!__inference__wrapped_model_918513
conv2d_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identityĚ
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpă
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ*
paddingVALID*
strides
2
sequential/conv2d/Conv2DĂ
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOpÓ
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2
sequential/conv2d/BiasAddĄ
sequential/activation/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2
sequential/activation/Relué
 sequential/max_pooling2d/MaxPoolMaxPool(sequential/activation/Relu:activations:0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolÓ
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2DÉ
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOpŰ
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
sequential/conv2d_1/BiasAddŠ
sequential/dropout/IdentityIdentity$sequential/conv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
sequential/dropout/Identity§
sequential/activation_1/ReluRelu$sequential/dropout/Identity:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
sequential/activation_1/Reluí
"sequential/max_pooling2d_1/MaxPoolMaxPool*sequential/activation_1/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙II*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPoolÓ
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOp
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*
paddingVALID*
strides
2
sequential/conv2d_2/Conv2DÉ
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOpŮ
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
sequential/conv2d_2/BiasAddŤ
sequential/dropout_1/IdentityIdentity$sequential/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
sequential/dropout_1/Identity§
sequential/activation_2/ReluRelu&sequential/dropout_1/Identity:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
sequential/activation_2/Reluí
"sequential/max_pooling2d_2/MaxPoolMaxPool*sequential/activation_2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙##*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPool
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙d 2
sequential/flatten/ConstÇ
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_2/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	2
sequential/flatten/ReshapeÂ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
É	*
dtype02(
&sequential/dense/MatMul/ReadVariableOpĂ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential/dense/MatMulż
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpĹ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential/dense/BiasAdd˘
sequential/activation_3/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential/activation_3/Sigmoidw
IdentityIdentity#sequential/activation_3/Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ:::::::::_ [
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
&
_user_specified_nameconv2d_input
ś5

F__inference_sequential_layer_call_and_return_conditional_losses_918821
conv2d_input
conv2d_918790
conv2d_918792
conv2d_1_918797
conv2d_1_918799
conv2d_2_918805
conv2d_2_918807
dense_918814
dense_918816
identity˘conv2d/StatefulPartitionedCall˘ conv2d_1/StatefulPartitionedCall˘ conv2d_2/StatefulPartitionedCall˘dense/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_918790conv2d_918792*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_9185632 
conv2d/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_9185842
activation/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_9185192
max_pooling2d/PartitionedCallÂ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_918797conv2d_1_918799*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_9186032"
 conv2d_1/StatefulPartitionedCall
dropout/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_9186362
dropout/PartitionedCall
activation_1/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_9186542
activation_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙II* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9185312!
max_pooling2d_1/PartitionedCallÂ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_918805conv2d_2_918807*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_9186732"
 conv2d_2/StatefulPartitionedCall
dropout_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_9187062
dropout_1/PartitionedCall
activation_2/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_9187242
activation_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙##* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9185432!
max_pooling2d_2/PartitionedCallř
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_9187392
flatten/PartitionedCall˘
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_918814dense_918816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9187572
dense/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_9187782
activation_3/PartitionedCall
IdentityIdentity%activation_3/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:_ [
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
&
_user_specified_nameconv2d_input
ż
_
C__inference_flatten_layer_call_and_return_conditional_losses_918739

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙d 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙##:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙##
 
_user_specified_nameinputs
úH
˘
__inference__traced_save_919400
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_7c008cd6ab564ec4aa958cd1d0ce3ffb/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameĆ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ř
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesĚ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ń
_input_shapesż
ź: :::::::
É	:: : : : : : : : : :::::::
É	::::::::
É	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
É	: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
É	: 

_output_shapes
::-)
'
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::& "
 
_output_shapes
:
É	: !

_output_shapes
::"

_output_shapes
: 
˙
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_918519

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
É
I
-__inference_activation_1_layer_call_fn_919182

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_9186542
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ż
D
(__inference_dropout_layer_call_fn_919172

inputs
identityĎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_9186362
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ú
d
H__inference_activation_2_layer_call_and_return_conditional_losses_918724

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙GG:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG
 
_user_specified_nameinputs
Ĺ
G
+__inference_activation_layer_call_fn_919126

inputs
identityŇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_9185842
PartitionedCallw
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2

Identity"
identityIdentity:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ
 
_user_specified_nameinputs
Ç
c
*__inference_dropout_1_layer_call_fn_919223

inputs
identity˘StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_9187012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙GG22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG
 
_user_specified_nameinputs
Đ
Š
A__inference_dense_layer_call_and_return_conditional_losses_918757

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
É	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*0
_input_shapes
:˙˙˙˙˙˙˙˙˙É	:::Q M
)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	
 
_user_specified_nameinputs
°
L
0__inference_max_pooling2d_2_layer_call_fn_918549

inputs
identityď
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9185432
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¸
Ú
+__inference_sequential_layer_call_fn_919076

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity˘StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9188582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
 
_user_specified_nameinputs

I
-__inference_activation_3_layer_call_fn_919278

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_9187782
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¸
d
H__inference_activation_3_layer_call_and_return_conditional_losses_918778

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ž
Ź
D__inference_conv2d_2_layer_call_and_return_conditional_losses_918673

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpĽ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙II:::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙II
 
_user_specified_nameinputs

Ů
$__inference_signature_wrapper_918963
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity˘StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_9185132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
&
_user_specified_nameconv2d_input
Ŕ)
Ç
F__inference_sequential_layer_call_and_return_conditional_losses_919055

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityŤ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpź
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ*
paddingVALID*
strides
2
conv2d/Conv2D˘
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2
conv2d/BiasAdd
activation/ReluReluconv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2
activation/ReluČ
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool˛
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpÚ
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv2d_1/Conv2D¨
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpŻ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
conv2d_1/BiasAdd
dropout/IdentityIdentityconv2d_1/BiasAdd:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/Identity
activation_1/ReluReludropout/Identity:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
activation_1/ReluĚ
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙II*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool˛
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÚ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*
paddingVALID*
strides
2
conv2d_2/Conv2D¨
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp­
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
conv2d_2/BiasAdd
dropout_1/IdentityIdentityconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout_1/Identity
activation_2/ReluReludropout_1/Identity:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
activation_2/ReluĚ
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙##*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙d 2
flatten/Const
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	2
flatten/ReshapeĄ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
É	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/BiasAdd
activation_3/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
activation_3/Sigmoidl
IdentityIdentityactivation_3/Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ:::::::::Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
 
_user_specified_nameinputs
°8
Ö
F__inference_sequential_layer_call_and_return_conditional_losses_918787
conv2d_input
conv2d_918574
conv2d_918576
conv2d_1_918614
conv2d_1_918616
conv2d_2_918684
conv2d_2_918686
dense_918768
dense_918770
identity˘conv2d/StatefulPartitionedCall˘ conv2d_1/StatefulPartitionedCall˘ conv2d_2/StatefulPartitionedCall˘dense/StatefulPartitionedCall˘dropout/StatefulPartitionedCall˘!dropout_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_918574conv2d_918576*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_9185632 
conv2d/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_9185842
activation/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_9185192
max_pooling2d/PartitionedCallÂ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_918614conv2d_1_918616*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_9186032"
 conv2d_1/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_9186312!
dropout/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_9186542
activation_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙II* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9185312!
max_pooling2d_1/PartitionedCallÂ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_918684conv2d_2_918686*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_9186732"
 conv2d_2/StatefulPartitionedCallŔ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_9187012#
!dropout_1/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_9187242
activation_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙##* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9185432!
max_pooling2d_2/PartitionedCallř
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_9187392
flatten/PartitionedCall˘
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_918768dense_918770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9187572
dense/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_9187782
activation_3/PartitionedCallĆ
IdentityIdentity%activation_3/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:_ [
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
&
_user_specified_nameconv2d_input
â
d
H__inference_activation_1_layer_call_and_return_conditional_losses_919177

inputs
identityY
ReluReluinputs*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Á
I
-__inference_activation_2_layer_call_fn_919238

inputs
identityŇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_9187242
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙GG:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG
 
_user_specified_nameinputs
¸
Ź
D__inference_conv2d_1_layer_call_and_return_conditional_losses_918603

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2	
BiasAddo
IdentityIdentityBiasAdd:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙:::Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ë
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_918701

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape˝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yÇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙GG:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG
 
_user_specified_nameinputs
ł
Ş
B__inference_conv2d_layer_call_and_return_conditional_losses_919107

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2	
BiasAddo
IdentityIdentityBiasAdd:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙ŹŹ:::Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
 
_user_specified_nameinputs
Ę
ŕ
+__inference_sequential_layer_call_fn_918877
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity˘StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9188582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
&
_user_specified_nameconv2d_input
ť
F
*__inference_dropout_1_layer_call_fn_919228

inputs
identityĎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_9187062
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙GG:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG
 
_user_specified_nameinputs

~
)__inference_conv2d_2_layer_call_fn_919201

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_9186732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙II::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙II
 
_user_specified_nameinputs
Ý
{
&__inference_dense_layer_call_fn_919268

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9187572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*0
_input_shapes
:˙˙˙˙˙˙˙˙˙É	::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	
 
_user_specified_nameinputs

~
)__inference_conv2d_1_layer_call_fn_919145

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_9186032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ň
a
C__inference_dropout_layer_call_and_return_conditional_losses_918636

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ż
_
C__inference_flatten_layer_call_and_return_conditional_losses_919244

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙d 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙##:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙##
 
_user_specified_nameinputs
ł
Ş
B__inference_conv2d_layer_call_and_return_conditional_losses_918563

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2	
BiasAddo
IdentityIdentityBiasAdd:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙ŹŹ:::Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
 
_user_specified_nameinputs
˘
Ö
"__inference__traced_restore_919509
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1,
(assignvariableop_17_adam_conv2d_kernel_m*
&assignvariableop_18_adam_conv2d_bias_m.
*assignvariableop_19_adam_conv2d_1_kernel_m,
(assignvariableop_20_adam_conv2d_1_bias_m.
*assignvariableop_21_adam_conv2d_2_kernel_m,
(assignvariableop_22_adam_conv2d_2_bias_m+
'assignvariableop_23_adam_dense_kernel_m)
%assignvariableop_24_adam_dense_bias_m,
(assignvariableop_25_adam_conv2d_kernel_v*
&assignvariableop_26_adam_conv2d_bias_v.
*assignvariableop_27_adam_conv2d_1_kernel_v,
(assignvariableop_28_adam_conv2d_1_bias_v.
*assignvariableop_29_adam_conv2d_2_kernel_v,
(assignvariableop_30_adam_conv2d_2_bias_v+
'assignvariableop_31_adam_dense_kernel_v)
%assignvariableop_32_adam_dense_bias_v
identity_34˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_31˘AssignVariableOp_32˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9Ě
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ř
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesŇ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesŘ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ł
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ľ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ľ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7˘
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8Ą
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ł
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10§
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ś
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ž
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ą
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ą
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ł
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ł
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17°
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_conv2d_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ž
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_conv2d_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19˛
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_conv2d_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_conv2d_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21˛
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ż
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24­
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25°
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_conv2d_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ž
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_conv2d_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27˛
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28°
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29˛
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ż
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32­
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp´
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33§
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*
_input_shapes
: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Š
D
(__inference_flatten_layer_call_fn_919249

inputs
identityĆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_9187392
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙##:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙##
 
_user_specified_nameinputs
â
d
H__inference_activation_1_layer_call_and_return_conditional_losses_918654

inputs
identityY
ReluReluinputs*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_918543

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ů
b
C__inference_dropout_layer_call_and_return_conditional_losses_918631

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeż
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yÉ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ś<
Ç
F__inference_sequential_layer_call_and_return_conditional_losses_919016

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityŤ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpź
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ*
paddingVALID*
strides
2
conv2d/Conv2D˘
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2
conv2d/BiasAdd
activation/ReluReluconv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2
activation/ReluČ
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool˛
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpÚ
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv2d_1/Conv2D¨
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpŻ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
conv2d_1/BiasAdds
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/ConstŠ
dropout/dropout/MulMulconv2d_1/BiasAdd:output:0dropout/dropout/Const:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/dropout/Mulw
dropout/dropout/ShapeShapeconv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape×
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yé
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/dropout/GreaterEqual˘
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/dropout/CastĽ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/dropout/Mul_1
activation_1/ReluReludropout/dropout/Mul_1:z:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
activation_1/ReluĚ
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙II*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool˛
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÚ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*
paddingVALID*
strides
2
conv2d_2/Conv2D¨
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp­
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
conv2d_2/BiasAddw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Const­
dropout_1/dropout/MulMulconv2d_2/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout_1/dropout/Mul{
dropout_1/dropout/ShapeShapeconv2d_2/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeŰ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yď
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2 
dropout_1/dropout/GreaterEqualŚ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout_1/dropout/CastŤ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout_1/dropout/Mul_1
activation_2/ReluReludropout_1/dropout/Mul_1:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
activation_2/ReluĚ
max_pooling2d_2/MaxPoolMaxPoolactivation_2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙##*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙d 2
flatten/Const
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	2
flatten/ReshapeĄ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
É	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense/BiasAdd
activation_3/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
activation_3/Sigmoidl
IdentityIdentityactivation_3/Sigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ:::::::::Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
 
_user_specified_nameinputs
¸
Ź
D__inference_conv2d_1_layer_call_and_return_conditional_losses_919136

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2	
BiasAddo
IdentityIdentityBiasAdd:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙:::Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Đ
Š
A__inference_dense_layer_call_and_return_conditional_losses_919259

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
É	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*0
_input_shapes
:˙˙˙˙˙˙˙˙˙É	:::Q M
)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	
 
_user_specified_nameinputs
Ë
a
(__inference_dropout_layer_call_fn_919167

inputs
identity˘StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_9186312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ů
b
C__inference_dropout_layer_call_and_return_conditional_losses_919157

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const~
dropout/MulMulinputsdropout/Const:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeż
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yÉ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2
dropout/Mul_1p
IdentityIdentitydropout/Mul_1:z:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ë
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_919213

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape˝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yÇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙GG:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG
 
_user_specified_nameinputs
°
L
0__inference_max_pooling2d_1_layer_call_fn_918537

inputs
identityď
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9185312
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¸
d
H__inference_activation_3_layer_call_and_return_conditional_losses_919273

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ě
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_919218

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙GG:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG
 
_user_specified_nameinputs
Ę
ŕ
+__inference_sequential_layer_call_fn_918932
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity˘StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9189132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
&
_user_specified_nameconv2d_input
Ź
J
.__inference_max_pooling2d_layer_call_fn_918525

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_9185192
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ň
a
C__inference_dropout_layer_call_and_return_conditional_losses_919162

inputs

identity_1e
IdentityIdentityinputs*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identityt

Identity_1IdentityIdentity:output:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙2

Identity_1"!

identity_1Identity_1:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

|
'__inference_conv2d_layer_call_fn_919116

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_9185632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙ŹŹ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
 
_user_specified_nameinputs
ŕ
b
F__inference_activation_layer_call_and_return_conditional_losses_919121

inputs
identityY
ReluReluinputs*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2
Reluq
IdentityIdentityRelu:activations:0*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ2

Identity"
identityIdentity:output:0*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ:Z V
2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ
 
_user_specified_nameinputs
8
Đ
F__inference_sequential_layer_call_and_return_conditional_losses_918858

inputs
conv2d_918827
conv2d_918829
conv2d_1_918834
conv2d_1_918836
conv2d_2_918842
conv2d_2_918844
dense_918851
dense_918853
identity˘conv2d/StatefulPartitionedCall˘ conv2d_1/StatefulPartitionedCall˘ conv2d_2/StatefulPartitionedCall˘dense/StatefulPartitionedCall˘dropout/StatefulPartitionedCall˘!dropout_1/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_918827conv2d_918829*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_9185632 
conv2d/StatefulPartitionedCall
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙ŞŞ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_9185842
activation/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_9185192
max_pooling2d/PartitionedCallÂ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_918834conv2d_1_918836*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_9186032"
 conv2d_1/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_9186312!
dropout/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_9186542
activation_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙II* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_9185312!
max_pooling2d_1/PartitionedCallÂ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_918842conv2d_2_918844*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_9186732"
 conv2d_2/StatefulPartitionedCallŔ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_9187012#
!dropout_1/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙GG* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_9187242
activation_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙##* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_9185432!
max_pooling2d_2/PartitionedCallř
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:˙˙˙˙˙˙˙˙˙É	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_9187392
flatten/PartitionedCall˘
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_918851dense_918853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9187572
dense/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_9187782
activation_3/PartitionedCallĆ
IdentityIdentity%activation_3/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:˙˙˙˙˙˙˙˙˙ŹŹ::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Y U
1
_output_shapes
:˙˙˙˙˙˙˙˙˙ŹŹ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_918531

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ă
serving_defaultŻ
O
conv2d_input?
serving_default_conv2d_input:0˙˙˙˙˙˙˙˙˙ŹŹ@
activation_30
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:ą÷
űQ
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
layer-13
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
Ŕ__call__
Á_default_save_signature
+Â&call_and_return_all_conditional_losses"N
_tf_keras_sequentialěM{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ů


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
Ă__call__
+Ä&call_and_return_all_conditional_losses"Ň	
_tf_keras_layer¸	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 300, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 300, 1]}}
Ó
regularization_losses
trainable_variables
	variables
	keras_api
Ĺ__call__
+Ć&call_and_return_all_conditional_losses"Â
_tf_keras_layer¨{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
ý
regularization_losses
 trainable_variables
!	variables
"	keras_api
Ç__call__
+Č&call_and_return_all_conditional_losses"ě
_tf_keras_layerŇ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ü	

#kernel
$bias
%regularization_losses
&trainable_variables
'	variables
(	keras_api
É__call__
+Ę&call_and_return_all_conditional_losses"Ő
_tf_keras_layerť{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 149, 149, 128]}}
ă
)regularization_losses
*trainable_variables
+	variables
,	keras_api
Ë__call__
+Ě&call_and_return_all_conditional_losses"Ň
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
×
-regularization_losses
.trainable_variables
/	variables
0	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"Ć
_tf_keras_layerŹ{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}

1regularization_losses
2trainable_variables
3	variables
4	keras_api
Ď__call__
+Đ&call_and_return_all_conditional_losses"đ
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ú	

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
Ń__call__
+Ň&call_and_return_all_conditional_losses"Ó
_tf_keras_layerš{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 73, 73, 128]}}
ç
;regularization_losses
<trainable_variables
=	variables
>	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"Ö
_tf_keras_layerź{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
×
?regularization_losses
@trainable_variables
A	variables
B	keras_api
Ő__call__
+Ö&call_and_return_all_conditional_losses"Ć
_tf_keras_layerŹ{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}

Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
×__call__
+Ř&call_and_return_all_conditional_losses"đ
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ä
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
Ů__call__
+Ú&call_and_return_all_conditional_losses"Ó
_tf_keras_layerš{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
÷

Kkernel
Lbias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
Ű__call__
+Ü&call_and_return_all_conditional_losses"Đ
_tf_keras_layerś{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 156800}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 156800]}}
Ú
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
Ý__call__
+Ţ&call_and_return_all_conditional_losses"É
_tf_keras_layerŻ{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}
ó
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratem°mą#m˛$mł5m´6mľKmśLmˇv¸vš#vş$vť5vź6v˝KvžLvż"
	optimizer
 "
trackable_list_wrapper
X
0
1
#2
$3
54
65
K6
L7"
trackable_list_wrapper
X
0
1
#2
$3
54
65
K6
L7"
trackable_list_wrapper
Î
regularization_losses
trainable_variables
	variables
Zmetrics
[layer_metrics

\layers
]non_trainable_variables
^layer_regularization_losses
Ŕ__call__
Á_default_save_signature
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
-
ßserving_default"
signature_map
(:&2conv2d/kernel
:2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
regularization_losses
trainable_variables
	variables
_metrics
`layer_metrics

alayers
bnon_trainable_variables
clayer_regularization_losses
Ă__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
regularization_losses
trainable_variables
	variables
dmetrics
elayer_metrics

flayers
gnon_trainable_variables
hlayer_regularization_losses
Ĺ__call__
+Ć&call_and_return_all_conditional_losses
'Ć"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
regularization_losses
 trainable_variables
!	variables
imetrics
jlayer_metrics

klayers
lnon_trainable_variables
mlayer_regularization_losses
Ç__call__
+Č&call_and_return_all_conditional_losses
'Č"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_1/kernel
:2conv2d_1/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
°
%regularization_losses
&trainable_variables
'	variables
nmetrics
olayer_metrics

players
qnon_trainable_variables
rlayer_regularization_losses
É__call__
+Ę&call_and_return_all_conditional_losses
'Ę"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
)regularization_losses
*trainable_variables
+	variables
smetrics
tlayer_metrics

ulayers
vnon_trainable_variables
wlayer_regularization_losses
Ë__call__
+Ě&call_and_return_all_conditional_losses
'Ě"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
-regularization_losses
.trainable_variables
/	variables
xmetrics
ylayer_metrics

zlayers
{non_trainable_variables
|layer_regularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
˛
1regularization_losses
2trainable_variables
3	variables
}metrics
~layer_metrics

layers
non_trainable_variables
 layer_regularization_losses
Ď__call__
+Đ&call_and_return_all_conditional_losses
'Đ"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_2/kernel
:2conv2d_2/bias
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
ľ
7regularization_losses
8trainable_variables
9	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
Ń__call__
+Ň&call_and_return_all_conditional_losses
'Ň"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
;regularization_losses
<trainable_variables
=	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
?regularization_losses
@trainable_variables
A	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
Ő__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Cregularization_losses
Dtrainable_variables
E	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
×__call__
+Ř&call_and_return_all_conditional_losses
'Ř"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Gregularization_losses
Htrainable_variables
I	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
Ů__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
 :
É	2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
ľ
Mregularization_losses
Ntrainable_variables
O	variables
metrics
layer_metrics
layers
non_trainable_variables
 layer_regularization_losses
Ű__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ľ
Qregularization_losses
Rtrainable_variables
S	variables
 metrics
Ąlayer_metrics
˘layers
Łnon_trainable_variables
 ¤layer_regularization_losses
Ý__call__
+Ţ&call_and_return_all_conditional_losses
'Ţ"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
Ľ0
Ś1"
trackable_list_wrapper
 "
trackable_dict_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ż

§total

¨count
Š	variables
Ş	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
˙

Ťtotal

Źcount
­
_fn_kwargs
Ž	variables
Ż	keras_api"ł
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
0
§0
¨1"
trackable_list_wrapper
.
Š	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ť0
Ź1"
trackable_list_wrapper
.
Ž	variables"
_generic_user_object
-:+2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
0:.2Adam/conv2d_1/kernel/m
!:2Adam/conv2d_1/bias/m
0:.2Adam/conv2d_2/kernel/m
!:2Adam/conv2d_2/bias/m
%:#
É	2Adam/dense/kernel/m
:2Adam/dense/bias/m
-:+2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
0:.2Adam/conv2d_1/kernel/v
!:2Adam/conv2d_1/bias/v
0:.2Adam/conv2d_2/kernel/v
!:2Adam/conv2d_2/bias/v
%:#
É	2Adam/dense/kernel/v
:2Adam/dense/bias/v
ú2÷
+__inference_sequential_layer_call_fn_919097
+__inference_sequential_layer_call_fn_919076
+__inference_sequential_layer_call_fn_918877
+__inference_sequential_layer_call_fn_918932Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
î2ë
!__inference__wrapped_model_918513Ĺ
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *5˘2
0-
conv2d_input˙˙˙˙˙˙˙˙˙ŹŹ
ć2ă
F__inference_sequential_layer_call_and_return_conditional_losses_919055
F__inference_sequential_layer_call_and_return_conditional_losses_918787
F__inference_sequential_layer_call_and_return_conditional_losses_918821
F__inference_sequential_layer_call_and_return_conditional_losses_919016Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ń2Î
'__inference_conv2d_layer_call_fn_919116˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ě2é
B__inference_conv2d_layer_call_and_return_conditional_losses_919107˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_activation_layer_call_fn_919126˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
F__inference_activation_layer_call_and_return_conditional_losses_919121˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
.__inference_max_pooling2d_layer_call_fn_918525ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ą2Ž
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_918519ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ó2Đ
)__inference_conv2d_1_layer_call_fn_919145˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_conv2d_1_layer_call_and_return_conditional_losses_919136˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
(__inference_dropout_layer_call_fn_919167
(__inference_dropout_layer_call_fn_919172´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ä2Á
C__inference_dropout_layer_call_and_return_conditional_losses_919162
C__inference_dropout_layer_call_and_return_conditional_losses_919157´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
×2Ô
-__inference_activation_1_layer_call_fn_919182˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ň2ď
H__inference_activation_1_layer_call_and_return_conditional_losses_919177˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
0__inference_max_pooling2d_1_layer_call_fn_918537ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ł2°
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_918531ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ó2Đ
)__inference_conv2d_2_layer_call_fn_919201˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
î2ë
D__inference_conv2d_2_layer_call_and_return_conditional_losses_919192˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
*__inference_dropout_1_layer_call_fn_919223
*__inference_dropout_1_layer_call_fn_919228´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Č2Ĺ
E__inference_dropout_1_layer_call_and_return_conditional_losses_919218
E__inference_dropout_1_layer_call_and_return_conditional_losses_919213´
Ť˛§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
×2Ô
-__inference_activation_2_layer_call_fn_919238˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ň2ď
H__inference_activation_2_layer_call_and_return_conditional_losses_919233˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
2
0__inference_max_pooling2d_2_layer_call_fn_918549ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ł2°
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_918543ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ň2Ď
(__inference_flatten_layer_call_fn_919249˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_flatten_layer_call_and_return_conditional_losses_919244˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Đ2Í
&__inference_dense_layer_call_fn_919268˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_dense_layer_call_and_return_conditional_losses_919259˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
×2Ô
-__inference_activation_3_layer_call_fn_919278˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ň2ď
H__inference_activation_3_layer_call_and_return_conditional_losses_919273˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
8B6
$__inference_signature_wrapper_918963conv2d_inputŽ
!__inference__wrapped_model_918513#$56KL?˘<
5˘2
0-
conv2d_input˙˙˙˙˙˙˙˙˙ŹŹ
Ş ";Ş8
6
activation_3&#
activation_3˙˙˙˙˙˙˙˙˙ş
H__inference_activation_1_layer_call_and_return_conditional_losses_919177n:˘7
0˘-
+(
inputs˙˙˙˙˙˙˙˙˙
Ş "0˘-
&#
0˙˙˙˙˙˙˙˙˙
 
-__inference_activation_1_layer_call_fn_919182a:˘7
0˘-
+(
inputs˙˙˙˙˙˙˙˙˙
Ş "# ˙˙˙˙˙˙˙˙˙ś
H__inference_activation_2_layer_call_and_return_conditional_losses_919233j8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙GG
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙GG
 
-__inference_activation_2_layer_call_fn_919238]8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙GG
Ş "!˙˙˙˙˙˙˙˙˙GG¤
H__inference_activation_3_layer_call_and_return_conditional_losses_919273X/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 |
-__inference_activation_3_layer_call_fn_919278K/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙¸
F__inference_activation_layer_call_and_return_conditional_losses_919121n:˘7
0˘-
+(
inputs˙˙˙˙˙˙˙˙˙ŞŞ
Ş "0˘-
&#
0˙˙˙˙˙˙˙˙˙ŞŞ
 
+__inference_activation_layer_call_fn_919126a:˘7
0˘-
+(
inputs˙˙˙˙˙˙˙˙˙ŞŞ
Ş "# ˙˙˙˙˙˙˙˙˙ŞŞş
D__inference_conv2d_1_layer_call_and_return_conditional_losses_919136r#$:˘7
0˘-
+(
inputs˙˙˙˙˙˙˙˙˙
Ş "0˘-
&#
0˙˙˙˙˙˙˙˙˙
 
)__inference_conv2d_1_layer_call_fn_919145e#$:˘7
0˘-
+(
inputs˙˙˙˙˙˙˙˙˙
Ş "# ˙˙˙˙˙˙˙˙˙ś
D__inference_conv2d_2_layer_call_and_return_conditional_losses_919192n568˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙II
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙GG
 
)__inference_conv2d_2_layer_call_fn_919201a568˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙II
Ş "!˙˙˙˙˙˙˙˙˙GGˇ
B__inference_conv2d_layer_call_and_return_conditional_losses_919107q9˘6
/˘,
*'
inputs˙˙˙˙˙˙˙˙˙ŹŹ
Ş "0˘-
&#
0˙˙˙˙˙˙˙˙˙ŞŞ
 
'__inference_conv2d_layer_call_fn_919116d9˘6
/˘,
*'
inputs˙˙˙˙˙˙˙˙˙ŹŹ
Ş "# ˙˙˙˙˙˙˙˙˙ŞŞŁ
A__inference_dense_layer_call_and_return_conditional_losses_919259^KL1˘.
'˘$
"
inputs˙˙˙˙˙˙˙˙˙É	
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 {
&__inference_dense_layer_call_fn_919268QKL1˘.
'˘$
"
inputs˙˙˙˙˙˙˙˙˙É	
Ş "˙˙˙˙˙˙˙˙˙ˇ
E__inference_dropout_1_layer_call_and_return_conditional_losses_919213n<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙GG
p
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙GG
 ˇ
E__inference_dropout_1_layer_call_and_return_conditional_losses_919218n<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙GG
p 
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙GG
 
*__inference_dropout_1_layer_call_fn_919223a<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙GG
p
Ş "!˙˙˙˙˙˙˙˙˙GG
*__inference_dropout_1_layer_call_fn_919228a<˘9
2˘/
)&
inputs˙˙˙˙˙˙˙˙˙GG
p 
Ş "!˙˙˙˙˙˙˙˙˙GGš
C__inference_dropout_layer_call_and_return_conditional_losses_919157r>˘;
4˘1
+(
inputs˙˙˙˙˙˙˙˙˙
p
Ş "0˘-
&#
0˙˙˙˙˙˙˙˙˙
 š
C__inference_dropout_layer_call_and_return_conditional_losses_919162r>˘;
4˘1
+(
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "0˘-
&#
0˙˙˙˙˙˙˙˙˙
 
(__inference_dropout_layer_call_fn_919167e>˘;
4˘1
+(
inputs˙˙˙˙˙˙˙˙˙
p
Ş "# ˙˙˙˙˙˙˙˙˙
(__inference_dropout_layer_call_fn_919172e>˘;
4˘1
+(
inputs˙˙˙˙˙˙˙˙˙
p 
Ş "# ˙˙˙˙˙˙˙˙˙Ş
C__inference_flatten_layer_call_and_return_conditional_losses_919244c8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙##
Ş "'˘$

0˙˙˙˙˙˙˙˙˙É	
 
(__inference_flatten_layer_call_fn_919249V8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙##
Ş "˙˙˙˙˙˙˙˙˙É	î
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_918531R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ć
0__inference_max_pooling2d_1_layer_call_fn_918537R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙î
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_918543R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ć
0__inference_max_pooling2d_2_layer_call_fn_918549R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ě
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_918519R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ä
.__inference_max_pooling2d_layer_call_fn_918525R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ä
F__inference_sequential_layer_call_and_return_conditional_losses_918787z#$56KLG˘D
=˘:
0-
conv2d_input˙˙˙˙˙˙˙˙˙ŹŹ
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 Ä
F__inference_sequential_layer_call_and_return_conditional_losses_918821z#$56KLG˘D
=˘:
0-
conv2d_input˙˙˙˙˙˙˙˙˙ŹŹ
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ž
F__inference_sequential_layer_call_and_return_conditional_losses_919016t#$56KLA˘>
7˘4
*'
inputs˙˙˙˙˙˙˙˙˙ŹŹ
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ž
F__inference_sequential_layer_call_and_return_conditional_losses_919055t#$56KLA˘>
7˘4
*'
inputs˙˙˙˙˙˙˙˙˙ŹŹ
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
+__inference_sequential_layer_call_fn_918877m#$56KLG˘D
=˘:
0-
conv2d_input˙˙˙˙˙˙˙˙˙ŹŹ
p

 
Ş "˙˙˙˙˙˙˙˙˙
+__inference_sequential_layer_call_fn_918932m#$56KLG˘D
=˘:
0-
conv2d_input˙˙˙˙˙˙˙˙˙ŹŹ
p 

 
Ş "˙˙˙˙˙˙˙˙˙
+__inference_sequential_layer_call_fn_919076g#$56KLA˘>
7˘4
*'
inputs˙˙˙˙˙˙˙˙˙ŹŹ
p

 
Ş "˙˙˙˙˙˙˙˙˙
+__inference_sequential_layer_call_fn_919097g#$56KLA˘>
7˘4
*'
inputs˙˙˙˙˙˙˙˙˙ŹŹ
p 

 
Ş "˙˙˙˙˙˙˙˙˙Á
$__inference_signature_wrapper_918963#$56KLO˘L
˘ 
EŞB
@
conv2d_input0-
conv2d_input˙˙˙˙˙˙˙˙˙ŹŹ";Ş8
6
activation_3&#
activation_3˙˙˙˙˙˙˙˙˙