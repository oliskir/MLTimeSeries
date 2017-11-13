
#include <sys/types.h>  // For stat().
#include <sys/stat.h>   // For stat().

void createDir(string dir)
{
    struct stat sb;
    if (!(stat(dir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))) 
    {
        // create the directory
        string makedir = "mkdir -p " + dir;
        system(makedir.c_str());

        cout << "Created output directoy: " << dir << endl;
    }
}

void plotfit(TString fname, size_t year = 2010, size_t month = 3, size_t day = 24) {

    createDir("figures");

    cout << "File: " << fname << endl;
    string date = Form("%lu-%lu-%lu", year, month, day);
    cout << "Show: " << date << endl;

    TFile f(fname, "READ");
    TTree * t = (TTree*)f.Get("heat");

    TCut selectDay = Form("year==%lu && month==%lu && day==%lu", year, month, day);

    int n0 = t->Draw("data:hour", selectDay, "goff");
    TGraph *g0 = new TGraph(n0, t->GetV2(), t->GetV1());
    
    int n1 = t->Draw("forecast:hour", selectDay && "forecast != 0", "goff");
    TGraph *g1 = new TGraph(n1, t->GetV2(), t->GetV1());

    TCanvas * c1 = new TCanvas("c1");

    auto mg = new TMultiGraph();

    mg->Add(g0, "APL"); 
    g0->SetTitle("Data"); 
    g0->SetLineWidth(1);  
    g0->SetLineColor(1);
    g0->SetLineStyle(3);
    g0->SetMarkerStyle(20);
    g0->SetMarkerColor(1);
    g0->SetFillStyle(0);

    mg->Add(g1, "L"); 
    g1->SetTitle("Forecast");
    g1->SetLineWidth(2);  
    g1->SetLineColor(2);
    g1->SetFillStyle(0);

    mg->SetTitle(Form("%s; Hour; Power (MW)", date.c_str()));

    mg->Draw("AP L");

    gPad->SetTicks();
//    c1->BuildLegend();
    
    c1->Print(Form("figures/%s.png", date.c_str()));
}
