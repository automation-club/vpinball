// Bumper.h: Definition of the Bumper class
//
//////////////////////////////////////////////////////////////////////
#pragma once
#if !defined(AFX_BUMPER_H__9A202FF0_7FAE_49BF_AA4C_C01C692E6DD9__INCLUDED_)
#define AFX_BUMPER_H__9A202FF0_7FAE_49BF_AA4C_C01C692E6DD9__INCLUDED_

#include "resource.h"       // main symbols

class BumperData
{
public:
   Vertex2D m_vCenter;
   float m_radius;
   float m_threshold; // speed at which ball needs to hit to register a hit
   float m_force; // force the bumper kicks back with
   float m_heightScale;
   float m_orientation;
   float m_ringSpeed;
   U32 m_time_msec;
   TimerDataRoot m_tdr;
   char m_szCapMaterial[32];
   char m_szBaseMaterial[32];
   char m_szSkirtMaterial[32];
   char m_szSurface[MAXTOKEN];
   bool m_fCapVisible;
   bool m_fBaseVisible;
   bool m_fReflectionEnabled;
};

/////////////////////////////////////////////////////////////////////////////
// Bumper

class Bumper :
   //public CComObjectRootEx<CComSingleThreadModel>,
   public IDispatchImpl<IBumper, &IID_IBumper, &LIBID_VPinballLib>,
   //public ISupportErrorInfo,
   public CComObjectRoot,
   public CComCoClass<Bumper, &CLSID_Bumper>,
   public EventProxy<Bumper, &DIID_IBumperEvents>,
   public IConnectionPointContainerImpl<Bumper>,
   public IProvideClassInfo2Impl<&CLSID_Bumper, &DIID_IBumperEvents, &LIBID_VPinballLib>,
   public ISelect,
   public IEditable,
   public Hitable,
   public IScriptable,
   public IFireEvents,
   public IPerPropertyBrowsing // Ability to fill in dropdown in property browser
   //public EditableImpl<Bumper>
{
public:
   Bumper();
   ~Bumper();

   BEGIN_COM_MAP(Bumper)
      COM_INTERFACE_ENTRY(IDispatch)
      COM_INTERFACE_ENTRY(IBumper)
      //COM_INTERFACE_ENTRY(ISupportErrorInfo)
      COM_INTERFACE_ENTRY_IMPL(IConnectionPointContainer)
      COM_INTERFACE_ENTRY(IPerPropertyBrowsing)
      COM_INTERFACE_ENTRY(IProvideClassInfo)
      COM_INTERFACE_ENTRY(IProvideClassInfo2)
   END_COM_MAP()
   //DECLARE_NOT_AGGREGATABLE(Bumper)
   // Remove the comment from the line above if you don't want your object to
   // support aggregation.

   STANDARD_EDITABLE_DECLARES(Bumper, eItemBumper, BUMPER, 1)

      BEGIN_CONNECTION_POINT_MAP(Bumper)
         CONNECTION_POINT_ENTRY(DIID_IBumperEvents)
      END_CONNECTION_POINT_MAP()

      DECLARE_REGISTRY_RESOURCEID(IDR_BUMPER)

      // ISupportsErrorInfo
      STDMETHOD(InterfaceSupportsErrorInfo)(REFIID riid);

      virtual void MoveOffset(const float dx, const float dy);
      virtual void SetObjectPos();

      virtual void GetDialogPanes(Vector<PropertyPane> *pvproppane);

      // Multi-object manipulation
      virtual void GetCenter(Vertex2D * const pv) const;
      virtual void PutCenter(const Vertex2D * const pv);

      virtual void UpdatePropertyPanes();
      virtual void SetDefaultPhysics(bool fromMouseClick);
      virtual void ExportMesh(FILE *f);

      void WriteRegDefaults();

      PinTable *m_ptable;

      BumperData m_d;

      BumperHitCircle *m_pbumperhitcircle;

      bool m_fDisabled;

private:
   void RenderBase(RenderDevice *pd3dDevice, const Material * const baseMaterial);
   void RenderCap(RenderDevice *pd3dDevice, const Material * const capMaterial);
   void RenderSocket(RenderDevice *pd3dDevice, const Material * const baseMaterial);
   void UpdateRing(RenderDevice *pd3dDevice);
   void GenerateBaseMesh(Vertex3D_NoTex2 *buf);
   void GenerateSocketMesh(Vertex3D_NoTex2 *buf);
   void GenerateRingMesh(Vertex3D_NoTex2 *buf);
   void GenerateCapMesh(Vertex3D_NoTex2 *buf);

   VertexBuffer *baseVertexBuffer;
   IndexBuffer *baseIndexBuffer;

   VertexBuffer *socketVertexBuffer;
   IndexBuffer *socketIndexBuffer;

   VertexBuffer *ringVertexBuffer;
   IndexBuffer *ringIndexBuffer;

   VertexBuffer *capVertexBuffer;
   IndexBuffer *capIndexBuffer;

   Matrix3D fullMatrix;

   Vertex3D_NoTex2 *ringVertices;
   Texture baseTexture;
   Texture ringTexture;
   Texture capTexture;
   Texture socketTexture;
   Material ringMaterial;

   PropertyPane *m_propVisual;

   float   baseHeight;
   bool    ringDown;
   bool    ringAnimate;

   // IBumper
public:
   STDMETHOD(get_BaseMaterial)(/*[out, retval]*/ BSTR *pVal);
   STDMETHOD(put_BaseMaterial)(/*[in]*/ BSTR newVal);
   STDMETHOD(get_SkirtMaterial)(/*[out, retval]*/ BSTR *pVal);
   STDMETHOD(put_SkirtMaterial)(/*[in]*/ BSTR newVal);
   STDMETHOD(get_Surface)(/*[out, retval]*/ BSTR *pVal);
   STDMETHOD(put_Surface)(/*[in]*/ BSTR newVal);
   STDMETHOD(get_Y)(/*[out, retval]*/ float *pVal);
   STDMETHOD(put_Y)(/*[in]*/ float newVal);
   STDMETHOD(get_X)(/*[out, retval]*/ float *pVal);
   STDMETHOD(put_X)(/*[in]*/ float newVal);
   STDMETHOD(get_CapMaterial)(/*[out, retval]*/ BSTR *pVal);
   STDMETHOD(put_CapMaterial)(/*[in]*/ BSTR newVal);
   STDMETHOD(get_Threshold)(/*[out, retval]*/ float *pVal);
   STDMETHOD(put_Threshold)(/*[in]*/ float newVal);
   STDMETHOD(get_Force)(/*[out, retval]*/ float *pVal);
   STDMETHOD(put_Force)(/*[in]*/ float newVal);
   STDMETHOD(get_HeightScale)(/*[out, retval]*/ float *pVal);
   STDMETHOD(put_HeightScale)(/*[in]*/ float newVal);
   STDMETHOD(get_RingSpeed)(/*[out, retval]*/ float *pVal);
   STDMETHOD(put_RingSpeed)(/*[in]*/ float newVal);
   STDMETHOD(get_Orientation)(/*[out, retval]*/ float *pVal);
   STDMETHOD(put_Orientation)(/*[in]*/ float newVal);
   STDMETHOD(get_Radius)(/*[out, retval]*/ float *pVal);
   STDMETHOD(put_Radius)(/*[in]*/ float newVal);
   STDMETHOD(get_Disabled)(/*[out, retval]*/ VARIANT_BOOL *pVal);
   STDMETHOD(put_Disabled)(/*[in]*/ VARIANT_BOOL newVal);
   STDMETHOD(get_CapVisible)(/*[out, retval]*/ VARIANT_BOOL *pVal);
   STDMETHOD(put_CapVisible)(/*[in]*/ VARIANT_BOOL newVal);
   STDMETHOD(get_BaseVisible)(/*[out, retval]*/ VARIANT_BOOL *pVal);
   STDMETHOD(put_BaseVisible)(/*[in]*/ VARIANT_BOOL newVal);
   STDMETHOD(get_ReflectionEnabled)(/*[out, retval]*/ VARIANT_BOOL *pVal);
   STDMETHOD(put_ReflectionEnabled)(/*[in]*/ VARIANT_BOOL newVal);
   STDMETHOD(PlayHit)();
};

#endif // !defined(AFX_BUMPER_H__9A202FF0_7FAE_49BF_AA4C_C01C692E6DD9__INCLUDED_)
