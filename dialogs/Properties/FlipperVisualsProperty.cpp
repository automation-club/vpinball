#include "stdafx.h"
#include "Properties/FlipperVisualsProperty.h"
#include <WindowsX.h>

FlipperVisualsProperty::FlipperVisualsProperty(const VectorProtected<ISelect> *pvsel) : BasePropertyDialog(IDD_PROPFLIPPER_VISUALS, pvsel)
{
    m_rubberThicknessEdit.SetDialog(this);
    m_rubberOffsetHeightEdit.SetDialog(this);
    m_rubberWidthEdit.SetDialog(this);
    m_posXEdit.SetDialog(this);
    m_posYEdit.SetDialog(this);
    m_baseRadiusEdit.SetDialog(this);
    m_endRadiusEdit.SetDialog(this);
    m_lengthEdit.SetDialog(this);
    m_startAngleEdit.SetDialog(this);
    m_endAngleEdit.SetDialog(this);
    m_heightEdit.SetDialog(this);
    m_maxDifficultLengthEdit.SetDialog(this);
    m_imageCombo.SetDialog(this);
    m_materialCombo.SetDialog(this);
    m_rubberMaterialCombo.SetDialog(this);
    m_surfaceCombo.SetDialog(this);
}

void FlipperVisualsProperty::UpdateVisuals(const int dispid/*=-1*/)
{
    for (int i = 0; i < m_pvsel->Size(); i++)
    {
        if ((m_pvsel->ElementAt(i) == NULL) || (m_pvsel->ElementAt(i)->GetItemType() != eItemFlipper))
            continue;
        Flipper * const flipper = (Flipper *)m_pvsel->ElementAt(i);
        if (dispid == IDC_MATERIAL_COMBO2 || dispid == -1)
            PropertyDialog::UpdateMaterialComboBox(flipper->GetPTable()->GetMaterialList(), m_rubberMaterialCombo, flipper->m_d.m_szRubberMaterial);
        if (dispid == 18 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_rubberThicknessEdit, flipper->m_d.m_rubberthickness);
        if (dispid == 24 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_rubberOffsetHeightEdit, flipper->m_d.m_rubberheight);
        if (dispid == 25 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_rubberWidthEdit, flipper->m_d.m_rubberwidth);
        if (dispid == 13 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_posXEdit, flipper->m_d.m_Center.x);
        if (dispid == 14 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_posYEdit, flipper->m_d.m_Center.y);
        if (dispid == 1 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_baseRadiusEdit, flipper->m_d.m_BaseRadius);
        if (dispid == 2 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_endRadiusEdit, flipper->m_d.m_EndRadius);
        if (dispid == 3 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_lengthEdit, flipper->m_d.m_FlipperRadiusMax);
        if (dispid == 4 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_startAngleEdit, flipper->m_d.m_StartAngle);
        if (dispid == 7 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_endAngleEdit, flipper->m_d.m_EndAngle);
        if (dispid == 107 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_heightEdit, flipper->m_d.m_height);
        if (dispid == 111 || dispid == -1)
            PropertyDialog::SetFloatTextbox(m_maxDifficultLengthEdit, flipper->GetFlipperRadiusMin());
        if (dispid == 1502 || dispid == -1)
            PropertyDialog::UpdateSurfaceComboBox(flipper->GetPTable(), m_surfaceCombo, flipper->m_d.m_szSurface);
        if (dispid == IDC_FLIPPER_ENABLED || dispid == -1)
            PropertyDialog::SetCheckboxState(::GetDlgItem(GetHwnd(), IDC_FLIPPER_ENABLED), flipper->m_d.m_enabled);
        UpdateBaseVisuals(flipper, &flipper->m_d,dispid);
        //only show the first element on multi-select
        break;
    }
}

void FlipperVisualsProperty::UpdateProperties(const int dispid)
{
    for (int i = 0; i < m_pvsel->Size(); i++)
    {
        if ((m_pvsel->ElementAt(i) == NULL) || (m_pvsel->ElementAt(i)->GetItemType() != eItemFlipper))
            continue;
        Flipper * const flipper = (Flipper *)m_pvsel->ElementAt(i);
        switch (dispid)
        {
            case 1:
                CHECK_UPDATE_ITEM(flipper->m_d.m_BaseRadius, PropertyDialog::GetFloatTextbox(m_baseRadiusEdit), flipper);
                break;
            case 2:
                CHECK_UPDATE_ITEM(flipper->m_d.m_EndRadius, PropertyDialog::GetFloatTextbox(m_endRadiusEdit), flipper);
                break;
            case 3:
                CHECK_UPDATE_ITEM(flipper->m_d.m_FlipperRadiusMax, PropertyDialog::GetFloatTextbox(m_lengthEdit), flipper);
                break;
            case 4:
                CHECK_UPDATE_ITEM(flipper->m_d.m_StartAngle, PropertyDialog::GetFloatTextbox(m_startAngleEdit), flipper);
                break;
            case 7:
                CHECK_UPDATE_ITEM(flipper->m_d.m_EndAngle, PropertyDialog::GetFloatTextbox(m_endAngleEdit), flipper);
                break;
            case 13:
                CHECK_UPDATE_ITEM(flipper->m_d.m_Center.x, PropertyDialog::GetFloatTextbox(m_posXEdit), flipper);
                break;
            case 14:
                CHECK_UPDATE_ITEM(flipper->m_d.m_Center.y, PropertyDialog::GetFloatTextbox(m_posYEdit), flipper);
                break;
            case 18:
                CHECK_UPDATE_ITEM(flipper->m_d.m_rubberthickness, PropertyDialog::GetFloatTextbox(m_rubberThicknessEdit), flipper);
                break;
            case 24:
                CHECK_UPDATE_ITEM(flipper->m_d.m_rubberheight, PropertyDialog::GetFloatTextbox(m_rubberOffsetHeightEdit), flipper);
                break;
            case 25:
                CHECK_UPDATE_ITEM(flipper->m_d.m_rubberwidth, PropertyDialog::GetFloatTextbox(m_rubberWidthEdit), flipper);
                break;
            case 107:
                CHECK_UPDATE_ITEM(flipper->m_d.m_height, PropertyDialog::GetFloatTextbox(m_heightEdit), flipper);
                break;
            case 111:
                CHECK_UPDATE_VALUE_SETTER(flipper->SetFlipperRadiusMin, flipper->GetFlipperRadiusMin, PropertyDialog::GetFloatTextbox, m_maxDifficultLengthEdit, flipper);
                break;
            case 1502:
                CHECK_UPDATE_COMBO_TEXT(flipper->m_d.m_szSurface, m_surfaceCombo, flipper);
                break;
            case IDC_FLIPPER_ENABLED:
                CHECK_UPDATE_ITEM(flipper->m_d.m_enabled, PropertyDialog::GetCheckboxState(::GetDlgItem(GetHwnd(), dispid)), flipper);
                break;
            case IDC_MATERIAL_COMBO2:
                CHECK_UPDATE_COMBO_TEXT_STRING(flipper->m_d.m_szRubberMaterial, m_rubberMaterialCombo, flipper);
                break;

            default:
                UpdateBaseProperties(flipper, &flipper->m_d, dispid);
                break;
        }
        flipper->UpdateStatusBarInfo();
    }
    UpdateVisuals(dispid);
}

BOOL FlipperVisualsProperty::OnInitDialog()
{
    m_imageCombo.AttachItem(DISPID_Image);
    m_materialCombo.AttachItem(IDC_MATERIAL_COMBO);
    m_rubberMaterialCombo.AttachItem(IDC_MATERIAL_COMBO2);
    m_rubberThicknessEdit.AttachItem(18);
    m_rubberOffsetHeightEdit.AttachItem(24);
    m_rubberWidthEdit.AttachItem(25);
    m_posXEdit.AttachItem(13);
    m_posYEdit.AttachItem(14);
    m_baseRadiusEdit.AttachItem(1);
    m_endRadiusEdit.AttachItem(2);
    m_lengthEdit.AttachItem(3);
    m_startAngleEdit.AttachItem(4);
    m_endAngleEdit.AttachItem(7);
    m_heightEdit.AttachItem(107);
    m_maxDifficultLengthEdit.AttachItem(111);
    m_surfaceCombo.AttachItem(1502);

    m_baseImageCombo = &m_imageCombo;
    m_baseMaterialCombo = &m_materialCombo;
    m_hVisibleCheck = ::GetDlgItem(GetHwnd(), IDC_VISIBLE_CHECK);
    m_hReflectionEnabledCheck = ::GetDlgItem(GetHwnd(), IDC_REFLECT_ENABLED_CHECK);
    UpdateVisuals();
    m_resizer.Initialize(*this, CRect(0, 0, 0, 0));
    m_resizer.AddChild(GetDlgItem(IDC_STATIC1), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC2), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC3), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC4), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC5), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC6), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC7), topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC8), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC9), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC10), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC11), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC12), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC13), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC14), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC15), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC16), topleft, 0);
    m_resizer.AddChild(GetDlgItem(IDC_STATIC17), topleft, 0);
    m_resizer.AddChild(m_imageCombo, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_materialCombo, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_rubberMaterialCombo, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_rubberThicknessEdit, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_rubberOffsetHeightEdit, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_rubberWidthEdit, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_posXEdit, topleft, 0);
    m_resizer.AddChild(m_posYEdit, topright, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_baseRadiusEdit, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_endRadiusEdit, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_lengthEdit, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_startAngleEdit, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_endAngleEdit, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_heightEdit, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_maxDifficultLengthEdit, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_surfaceCombo, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_hVisibleCheck, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(m_hReflectionEnabledCheck, topleft, RD_STRETCH_WIDTH);
    m_resizer.AddChild(GetDlgItem(IDC_FLIPPER_ENABLED), topleft, 0);
    return TRUE;
}

INT_PTR FlipperVisualsProperty::DialogProc(UINT uMsg, WPARAM wParam, LPARAM lParam)
{
   m_resizer.HandleMessage(uMsg, wParam, lParam);
   return DialogProcDefault(uMsg, wParam, lParam);
}
