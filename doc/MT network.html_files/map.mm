<map version="0.9.0">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1334220933777" ID="ID_905581508" MODIFIED="1337613312363" TEXT="MT network">
<node CREATED="1334220957744" ID="ID_1519753213" MODIFIED="1337613598818" POSITION="right" TEXT="Neurons">
<node CREATED="1334220967635" ID="ID_375649472" MODIFIED="1334220970466" TEXT="exc"/>
<node CREATED="1334220970996" ID="ID_668324120" MODIFIED="1334220972143" TEXT="inh"/>
<node CREATED="1334220977718" ID="ID_84889021" MODIFIED="1337610260604" TEXT="neural parameters">
<node CREATED="1337606852193" ID="ID_1093551702" MODIFIED="1337606911990" TEXT="Tuning properties"/>
</node>
<node CREATED="1334221059842" ID="ID_550021201" MODIFIED="1337610262886" TEXT="Neuron model">
<node CREATED="1334221066841" ID="ID_1936840038" MODIFIED="1337613604684" TEXT="IF_cond_exp">
<icon BUILTIN="button_ok"/>
</node>
<node CREATED="1334221073436" ID="ID_1124208320" MODIFIED="1337613609876" TEXT="AdExp">
<icon BUILTIN="button_cancel"/>
</node>
<node CREATED="1337606943917" ID="ID_1451856029" MODIFIED="1337613613155" TEXT="Hodgkin-Huxley">
<icon BUILTIN="button_cancel"/>
</node>
</node>
</node>
<node CREATED="1334220994143" ID="ID_1166660463" MODIFIED="1337613614294" POSITION="left" TEXT="Connections">
<node CREATED="1334221001700" FOLDED="true" ID="ID_468705535" MODIFIED="1337613642540" TEXT="Learned connectivity">
<node CREATED="1334221006542" ID="ID_703521133" MODIFIED="1334221008633" TEXT="BCPNN"/>
<node CREATED="1334221008875" FOLDED="true" ID="ID_1141708476" MODIFIED="1337613622228" TEXT="modified STDP">
<icon BUILTIN="help"/>
<node CREATED="1334221029979" ID="ID_614905723" MODIFIED="1334221033255" TEXT="-&gt;bias?"/>
</node>
</node>
<node CREATED="1334221043060" FOLDED="true" ID="ID_390055756" MODIFIED="1337612738188" TEXT="Pre-computed connectivity">
<node CREATED="1337610474797" ID="ID_1604847318" MODIFIED="1337612057650" TEXT="w_sigma_x, w_sigma_y">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Width of the pre-computed connectivity profile
    </p>
    <p>
      
    </p>
    <p>
      x_i, y_i, ... are tuning properties of cell i
    </p>
    <p>
      
    </p>
    <p>
      sigma_* handles how accurate connections are to be drawn:
    </p>
    <p>
      source position (in 4D) + movement = target
    </p>
    <p>
      
    </p>
    <p>
      large w_sigma_*: broad (deviation from unaccelerated movements possible to predict)
    </p>
    <p>
      small w_sigma_*:
    </p>
    <p>
      deviation from unaccelerated movements become less likely, straight line movements preferred
    </p>
    <p>
      
    </p>
    <p>
      
    </p>
    <p>
      latency = np.sqrt((x0 - x1)**2 + (y0 - y1)**2) / np.sqrt(u0**2 + v0**2)
    </p>
    <p>
      p = .5 * np.exp(-((x0 + u0 * latency - x1)**2 + (y0 + v0 * latency - y1)**2) / (2 * sigma_x**2)) \
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;* np.exp(-((u0-u1)**2 + (v0 - v1)**2) / (2 * sigma_v**2))
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1337611498733" ID="ID_1835343929" MODIFIED="1337612224376" TEXT="p_thresh_connection">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      If p &lt;= p_thresh_connection: do not connect source to target cell
    </p>
    <p>
      
    </p>
    <p>
      --&gt; decides on number of connections and cross-talk in the network
    </p>
    <p>
      --&gt; network density &lt;=&gt; sparseness
    </p>
    <p>
      
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1337611513545" ID="ID_1954690424" MODIFIED="1337612725012" TEXT="p_to_w_scaling">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Translation factor from p --&gt; weight
    </p>
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1337606620551" ID="ID_455964126" MODIFIED="1337606625086" TEXT="Random connectivity"/>
<node CREATED="1337610207546" ID="ID_1177106049" MODIFIED="1337610227619" TEXT="Synapse dynamics">
<node CREATED="1337610219392" ID="ID_1252541491" MODIFIED="1337612708273" TEXT="Short-term plasticity"/>
<node CREATED="1337610230544" ID="ID_1045578593" MODIFIED="1337610234712" TEXT="Synapse types">
<node CREATED="1337610236267" ID="ID_1765568339" MODIFIED="1337610249413" TEXT="gExp"/>
<node CREATED="1337610286707" ID="ID_158361138" MODIFIED="1337610304345" TEXT="gAlpha"/>
</node>
</node>
</node>
<node CREATED="1337606635797" ID="ID_1107849948" MODIFIED="1337613489404" POSITION="right" TEXT="Input">
<node CREATED="1337606642209" ID="ID_395731526" MODIFIED="1337610261580" TEXT="Motion energy">
<node CREATED="1337606702496" HGAP="35" ID="ID_600110310" MODIFIED="1337613516491" TEXT="Tuning properties">
<arrowlink DESTINATION="ID_1093551702" ENDARROW="Default" ENDINCLINATION="181;0;" ID="Arrow_ID_1746643199" STARTARROW="None" STARTINCLINATION="181;0;"/>
<icon BUILTIN="yes"/>
<node CREATED="1337611069429" ID="ID_1914233030" MODIFIED="1337613514428" TEXT="blur_X, blur_V">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      blur parameter decides how specific the tuning of cell properties are:
    </p>
    <p>
      if blur is large: responses to stimulus are unspecific (more uniform across the population)
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;small:&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;specific (individual tuning properties get crucial)
    </p>
    <p>
      
    </p>
    <p>
      for each time t:
    </p>
    <p>
      &#160;&#160;&#160;&#160;calculate position of stimulus ---&gt; x, y
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;L[cell] = np.exp( -.5 * (tuning_prop[cell, 0] - x)**2/blur_X**2
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;-.5 * (tuning_prop[cell, 1] - y)**2/blur_X**2
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;-.5 * (tuning_prop[cell, 2] - u0)**2/blur_V**2
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;-.5 * (tuning_prop[cell, 3] - v0)**2/blur_V**2
    </p>
    <p>
      &#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;)
    </p>
  </body>
</html></richcontent>
</node>
</node>
</node>
<node CREATED="1337606965397" FOLDED="true" ID="ID_606924616" MODIFIED="1337613597973" TEXT="Other cortical areas">
<node CREATED="1337610114708" ID="ID_1166165795" MODIFIED="1337610117182" TEXT="V1"/>
<node CREATED="1337610118089" ID="ID_362676413" MODIFIED="1337610119419" TEXT="V2"/>
<node CREATED="1337610120334" ID="ID_193074183" MODIFIED="1337610121769" TEXT="V3"/>
<node CREATED="1337610122435" ID="ID_310960856" MODIFIED="1337610123886" TEXT="V4"/>
<node CREATED="1337610140100" ID="ID_1305065767" MODIFIED="1337610141533" TEXT="FST"/>
<node CREATED="1337610142645" ID="ID_1050375428" MODIFIED="1337610143830" TEXT="MST"/>
</node>
</node>
</node>
</map>
