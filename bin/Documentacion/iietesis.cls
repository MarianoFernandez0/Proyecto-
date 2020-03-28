%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Clase de documento LaTeX para una tesis de posgrado del IIE.
%
% Version 1.0
% * Version inicial
% * Los integrantes del tribunal deben ser puestos a mano directamente
%   en iietesis.cls
%
%-- federico lecumberry
%    fefo@fing.edu.uy
%
% Version 1.1
% * Se agrega la opcion de proyecto de fin de carrera
% * Se crean comandos para integrantes del tribunal, etc
% * Se cambian los logos por versiones vectoriales
% * Gabriel Eirea, geirea@fing.edu.uy
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{iietesis}[2012/08/05 Clase de LaTeX para tesis del IIE,
FIng, UdelaR]
\LoadClassWithOptions{book}

\def\IIETesisVersion{1.1}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Packages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage[rightFloats,CaptionBefore]{fltpage} % 
\RequirePackage[font={footnotesize,sf}]{caption} %
\RequirePackage{subfig} %
\RequirePackage{graphicx} %
\RequirePackage[spanish]{babel} %
\RequirePackage[utf8]{inputenc} %
\RequirePackage{amsmath} %
\RequirePackage{amssymb} %
\RequirePackage{amsfonts} %
\RequirePackage[usenames,dvipsnames]{color} % added by me
\RequirePackage{multirow} % For multirow cells
\RequirePackage{booktabs} % For `professional' tables (toprule, etc.)
% \RequirePackage[T1]{fontenc}
% \RequirePackage[bitstream-charter]{mathdesign} %
% \RequirePackage{minion} %
\RequirePackage{url}
\RequirePackage{xspace} %
\RequirePackage{cite}
\RequirePackage{pifont} % To generate the circled numbers with \ding
\RequirePackage{afterpage} %
\RequirePackage{calc}
\RequirePackage{ifthen} %
\RequirePackage{makeidx}
\RequirePackage{totpages} %\RequirePackage{textcomp}
\RequirePackage[24hr, dayofweek, long]{datetime}
\RequirePackage{xifthen} % Provides \isempty test
% \RequirePackage{fullpage} %
\RequirePackage{fancyhdr} %
\RequirePackage{wasysym} % 
\RequirePackage{layout} %
\RequirePackage{ifthen} %
\RequirePackage[a4paper=true]{hyperref} % To follows url links

\RequirePackage{sectsty}
\allsectionsfont{\mdseries\sffamily}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Correcciones a babel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\addto\captionsspanish{%
    \def\prefacename{Prefacio}%
    \def\refname{Referencias}%
    \def\abstractname{Resumen}%
    % \def\bibname{Bibliograf\'{\i}a}%
    \def\bibname{Referencias}%
    \def\chaptername{Cap\'{\i}tulo}%
    \def\appendixname{Ap\'endice}%
    \def\contentsname{Tabla de contenidos}%
    \def\listfigurename{\'Indice de figuras}%
    \def\listtablename{\'Indice de tablas}%
    \def\indexname{\'Indice}%
    \def\figurename{Figura}%
    \def\tablename{Tabla}%
    \def\partname{Parte}%
    \def\pagename{P\'agina}%
}%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % Para tener una lista de referencias por capítulo, parte, etc. usar
%% % este paquete.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% \RequirePackage{chapterbib} %

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\DeclareOption{draft}{%
  \linespread{1}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Packages options.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
\newboolean{isMaster} 
\setboolean{isMaster}{false} % boolvar=true or false

\newboolean{isDoctorado}
\setboolean{isDoctorado}{false} % boolvar=true or false

\newboolean{isProyecto}
\setboolean{isProyecto}{false} % boolvar=true or false

\def\TesisProy{}
\def\TituloPosgrado{Elegir \texttt{master} o \texttt{doctor} o \texttt{proyecto}}
\def\ISSN{ISSN XXXX-XXXX}

\DeclareOption{master}{
  \def\TesisProy{Tesis}
  \def\TituloPosgrado{Magister en Ingenier\'{\i}a El\'ectrica}
  \def\ISSN{ISSN 1688-2806}
  \setboolean{isMaster}{true}
}

\DeclareOption{doctorado}{
  \def\TesisProy{Tesis}
  \def\TituloPosgrado{Doctor en Ingenier\'{\i}a El\'ectrica}
  \def\ISSN{ISSN 1688-2784}
  \setboolean{isDoctorado}{true}
}

\DeclareOption{proyecto}{
  \def\TesisProy{Memoria de proyecto}
  \def\TituloPosgrado{Ingeniero Electricista}
  \def\ISSN{}
  \setboolean{isProyecto}{true}
}

\ProcessOptions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% fancyheader configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%\fancypagestyle{plain}
\pagestyle{fancyplain}
%\addtolength{\headwidth}{2.6cm}
%\addtolength{\headwidth}{\marginparwidth}
%\addtolength{\headwidth}{-2mm}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\plainheadrulewidth}{0pt} %
\fancyhf{} %
\fancyfoot[LE,RO]{\fancyplain{}{\small\thepage}} %
\fancyhead[LE]{\fancyplain{}{\textsf{\nouppercase{\leftmark}}}}
\fancyhead[RO]{\fancyplain{}{\textsf{\nouppercase{\rightmark}}}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definir layout de la página
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\evensidemargin = 0cm%
\oddsidemargin = 2cm%
\addtolength{\voffset}{-0.5cm}
\addtolength{\textheight}{1cm}
% \addtolength{\headsep}{1.5cm}
  
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configuración de la fecha.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\dateseparator}{-}
\settimeformat{hhmmsstime}
\renewcommand{\sfdefault}{lmss}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Darle un poco más de espacio para figuras y cosas flotantes.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\topfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.75}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deshacer los cambios hechos por el comando \appendix en caso de
% tener varias partes y apéndices en cada una.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\unappendix}{\par
  % \setcounter{chapter}{0}%
  % \setcounter{section}{0}%
  \gdef\@chapapp{\chaptername}%
  \gdef\thechapter{\@arabic\c@chapter} 
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comandos
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\TutorNombreA}[1]{\def\@TutorNombreA{#1}}
\newcommand{\TutorAfiliacionA}[1]{\def\@TutorAfiliacionA{#1}}
\newcommand{\TutorNombreB}[1]{\def\@TutorNombreB{#1}}
\newcommand{\TutorAfiliacionB}[1]{\def\@TutorAfiliacionB{#1}}
\newcommand{\TribunalNombreA}[1]{\def\@TribunalNombreA{#1}}
\newcommand{\TribunalAfiliacionA}[1]{\def\@TribunalAfiliacionA{#1}}
\newcommand{\TribunalNombreB}[1]{\def\@TribunalNombreB{#1}}
\newcommand{\TribunalAfiliacionB}[1]{\def\@TribunalAfiliacionB{#1}}
\newcommand{\TribunalNombreC}[1]{\def\@TribunalNombreC{#1}}
\newcommand{\TribunalAfiliacionC}[1]{\def\@TribunalAfiliacionC{#1}}
\newcommand{\TribunalNombreD}[1]{\def\@TribunalNombreD{#1}}
\newcommand{\TribunalAfiliacionD}[1]{\def\@TribunalAfiliacionD{#1}}
\newcommand{\TribunalNombreE}[1]{\def\@TribunalNombreE{#1}}
\newcommand{\TribunalAfiliacionE}[1]{\def\@TribunalAfiliacionE{#1}}
\newcommand{\TribunalNombreF}[1]{\def\@TribunalNombreF{#1}}
\newcommand{\TribunalAfiliacionF}[1]{\def\@TribunalAfiliacionF{#1}}
\newcommand{\DAcademicoNombre}[1]{\def\@DAcademicoNombre{#1}}
\newcommand{\DAcademicoAfiliacion}[1]{\def\@DAcademicoAfiliacion{#1}}

\TutorNombreA{}
\TutorAfiliacionA{}
\TutorNombreB{}
\TutorAfiliacionB{}
\TribunalNombreA{}
\TribunalAfiliacionA{}
\TribunalNombreB{}
\TribunalAfiliacionB{}
\TribunalNombreC{}
\TribunalAfiliacionC{}
\TribunalNombreD{}
\TribunalAfiliacionD{}
\TribunalNombreE{}
\TribunalAfiliacionE{}
\TribunalNombreF{}
\TribunalAfiliacionF{}
\DAcademicoNombre{}
\DAcademicoAfiliacion{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definiciones
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\input{definitions}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Carátula  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\maketitle}[0]{%
  \thispagestyle{empty}\vbox to \vsize{%
    \vspace{-1cm}
    \begin{flushright}
      {\tiny\ISSN}
    \end{flushright}
    \vspace{.1cm}
    \begin{center}
      \begin{minipage}[t]{0.125\linewidth}
        \includegraphics[height=5em]{figuras/logo/logo_univ.pdf}
      \end{minipage}      
      \hfill
      \begin{minipage}[t]{0.73\linewidth}
        \begin{center}
          \vspace{-3em}
          \textsc{\small Universidad de la Rep\'ublica}\\
          \vspace{-0.2em}
          \textsc{\small Facultad de Ingenier\'{\i}a}
        \end{center}
      \end{minipage}      
      \hfill
      \begin{minipage}[t]{0.125\linewidth}
        \includegraphics[height=5em]{figuras/logo/logo_fing.pdf}
      \end{minipage}\\
    \end{center}
    \vspace{1.5em}
    \begin{center}
      {\huge \@title}
    \end{center}
    \vspace{1cm}
    \begin{center}
      \begin{minipage}[t]{0.9\linewidth}
        \begin{center}
          \textsc{\TesisProy{} presentada a la Facultad de Ingenier\'{\i}a de
            la Universidad de la Rep\'ublica
            por}\\\vspace{0.5cm}{\large
            \@author}\\\vspace{0.5cm}\textsc{en cumplimiento parcial
            de los requerimientos\\ para la obtenci\'on del t\'{\i}tulo
            de\\\TituloPosgrado{}.}
        \end{center}
      \end{minipage}
    \end{center}

    \vspace{2em}

    \begin{center}
      \textsc{\ifthenelse{\boolean{isProyecto}}{Tutor}{Director de Tesis}}
      \begin{minipage}[t]{0.9\linewidth}

        \@TutorNombreA %
        \dotfill{} %
        \@TutorAfiliacionA %

        \ifthenelse{\equal{\@TutorNombreB}{}}{}
                   {\@TutorNombreB \dotfill{} \@TutorAfiliacionB}

      \end{minipage}\\

      \vspace{2em}

      \textsc{Tribunal}\\
      \begin{minipage}[t]{0.9\linewidth}

        \@TribunalNombreA %
        \dotfill{} %
        \@TribunalAfiliacionA %

        \ifthenelse{\equal{\@TribunalNombreB}{}}{}
                   {\@TribunalNombreB \dotfill{} \@TribunalAfiliacionB}

        \ifthenelse{\equal{\@TribunalNombreC}{}}{}
                   {\@TribunalNombreC \dotfill{} \@TribunalAfiliacionC}

        \ifthenelse{\equal{\@TribunalNombreD}{}}{}
                   {\@TribunalNombreD \dotfill{} \@TribunalAfiliacionD}

        \ifthenelse{\equal{\@TribunalNombreE}{}}{}
                   {\@TribunalNombreE \dotfill{} \@TribunalAfiliacionE}

        \ifthenelse{\equal{\@TribunalNombreF}{}}{}
                   {\@TribunalNombreF \dotfill{} \@TribunalAfiliacionF}

      \end{minipage}\\
      
      \vspace{2em}

      \ifthenelse{\boolean{isProyecto}}{}{
      \textsc{Director Acad\'emico}\\
      \begin{minipage}[t]{0.9\linewidth}
        \@DAcademicoNombre % 
        \dotfill{} %
        \@DAcademicoAfiliacion %
      \end{minipage}\\%
      }
    \end{center}
    
    \vspace{3em}
    \vfill 

    \begin{center}
      \@location\\
      \@date
    \end{center}
  }%
  \newpage
  \infopage
  \cleardoublepage
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Redefinition of titlepage's parameters and their default values.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\date}[1]{\def\@date{#1}}
\renewcommand{\author}[1]{\def\@author{#1}}
\renewcommand{\title}[1]{\def\@title{#1}}
\newcommand{\location}[1]{\def\@location{#1}}
\newcommand{\blurb}[1]{\def\@blurb{#1}}
\newcommand{\email}[1]{\def\@email{#1}}
\date{\today}
\author{}
\title{}
\location{}
\blurb{}
\email{no email address}
%\makeatother

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% \newcommand*{\OrigChapter}{}
% \let\OrigChapter\chapter
% \newcounter{abstables}
% \newcounter{absfigures}
% \renewcommand*{\chapter}{%
%   \addtocounter{abstables}{\value{table}}%
%   \addtocounter{absfigures}{\value{figure}}%
%   \OrigChapter
% }
% \newcommand*{\AbsTables}{0}
% \makeatletter
% \AtBeginDocument{%
%   \AtEndDocument{%
%     \addtocounter{abstables}{\value{table}}%
%     \immediate\write\@mainaux{%
%       \string\gdef\string\AbsTables{\number\value{abstables}}%
%     }%
%   }%
% }
% \makeatother
% \newcommand*{\AbsFigures}{0}
% \makeatletter
% \AtBeginDocument{%
%   \AtEndDocument{%
%     \addtocounter{absfigures}{\value{figure}}%
%     \immediate\write\@mainaux{%
%       \string\gdef\string\AbsFigures{\number\value{absfigures}}%
%     } %
%   } %
% }
% \makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Eliminar estilo en las páginas en blanco
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\makeatletter
\def\cleardoublepage{
  \clearpage %
  \if@twoside %
  \ifodd %
  \c@page %
  \else
  \hbox{} %
  \thispagestyle{empty} %
  \begin{center}
    \vfill
    Esta p\'agina ha sido intencionalmente dejada en blanco.
    \vfill
  \end{center}
  \newpage %
  \if@twocolumn %
  \hbox{} %
  \begin{center}
    \vfill
    Esta p\'agina ha sido intencionalmente dejada en blanco.
    \vfill
  \end{center}
  \newpage %
  \fi
  \fi
  \fi
}
\makeatother

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\
% Information page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\
\newcommand{\infopage}{
  \newpage
  \thispagestyle{empty}
  \small
  \noindent\textit{\MyTitle}, \MyAuthor .\\\\
  \noindent\ISSN\\\\
  \noindent Esta tesis fue preparada en \LaTeX{} usando la clase iietesis (v\IIETesisVersion).\\
  Contiene un total de \ref{TotPages} p\'aginas.\\
  Compilada el \today{}.\\
  \url{\DocumentURL}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\
% Last page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\
\newcommand{\lastpage}{
  
  % \layout 
  \newpage
  \thispagestyle{empty}
  \begin{minipage}[b]{8cm}
    \begin{center}
    \end{center}
  \end{minipage}
  \newpage
  \thispagestyle{empty}
  \vspace*{0.4\textheight}
  % \begin{minipage}[b]{8cm}
  \begin{center}
    \small
    Esta es la \'ultima p\'agina.\\
    Compilado el \today{}.\\
    \url{\DocumentURL}
  \end{center}
  % \end{minipage}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\documenturl}[1]{%
  \newcommand{\DocumentURL}{#1}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand\OriginalAuthor{}
\let\OriginalAuthor\author
\renewcommand*{\author}[1]{%
  \OriginalAuthor{#1}%
  \newcommand*{\MyAuthor}{#1}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand\OriginalTitle{}
\let\OriginalTitle\title
\renewcommand*{\title}[1]{%
  \OriginalTitle{#1}%
  \newcommand*{\MyTitle}{#1}%
}

